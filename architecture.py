import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin, Sequential as Seq
from gcn_lib.sparse import MultiSeq, MLP, GraphConv, ResGraphBlock, DenseGraphBlock
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import TopKPooling
# from torch_geometric.utils import scatter_
from torchvision.models import resnet18, resnet34, resnet50, densenet121, mobilenet_v2
import efficientnet_pytorch as efficientnet

import torch.nn as nn
from torch_geometric.nn import DenseGINConv
# from torch_geometric.nn import DenseSAGEConv
# from torch_geometric.utils import to_dense_batch, to_dense_adj
from utils.geometric_util import DenseSAGEConv, to_dense_adj, to_dense_batch, scatter_
from torch.nn import Linear, LSTM
EPS = 1e-15
import pdb


class DenseJK(nn.Module):
    def __init__(self, mode, channels=None, num_layers=None):
        super(DenseJK, self).__init__()
        self.channel = channels
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm']

        if mode == 'lstm':
            assert channels is not None
            assert num_layers is not None
            self.lstm = LSTM(
                channels,
                channels  * num_layers // 2,
                bidirectional=True,
                batch_first=True)
            self.att = Linear(2 * channels * num_layers // 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lstm'):
            self.lstm.reset_parameters()
        if hasattr(self, 'att'):
            self.att.reset_parameters()

    def forward(self, xs):
        r"""Aggregates representations across different layers.

        Args:
            xs  [batch, nodes, featdim*3]
        """

        xs = torch.split(xs, self.channel, -1)# list of batch, node, featdim
        xs = torch.stack(xs,2)#[batch, nodes, num_layers, num_channels]
        shape = xs.shape
        x = xs.reshape((-1,shape[2],shape[3]))  # [ngraph * num_nodes , num_layers, num_channels]
        alpha, _ = self.lstm(x)
        alpha = self.att(alpha).squeeze(-1)  # [ngraph * num_nodes, num_layers]
        alpha = torch.softmax(alpha, dim=-1)
        x =  (x * alpha.unsqueeze(-1)).sum(dim=1)
        x = x.reshape((shape[0],shape[1],shape[3]))
        return x

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mode)

class GNN_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim,embedding_dim, bias, bn, add_loop, lin = True,gcn_name='SAGE', sync = False
                 ,activation = 'relu', jk = False):
        super(GNN_Module, self).__init__()
        self.jk = jk
        self.add_loop = add_loop
        self.gcn1 = self._gcn(gcn_name,input_dim, hidden_dim,bias, activation)#DenseSAGEConv(input_dim, hidden_dim, normalize= True, bias= bias)
        self.active1 = self._activation(activation)
        self.gcn2 = self._gcn(gcn_name,hidden_dim, hidden_dim,bias, activation)#DenseSAGEConv(hidden_dim, hidden_dim, normalize= True, bias= bias)
        self.active2 = self._activation(activation)
        self.gcn3 = self._gcn(gcn_name,hidden_dim, embedding_dim,bias, activation)#DenseSAGEConv(hidden_dim, embedding_dim, normalize=True, bias= bias)
        self.active3 = self._activation(activation)
        if bn:
            if sync:
                self.bn1 = nn.SyncBatchNorm(hidden_dim)
            else:
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.bn2 = nn.BatchNorm1d(hidden_dim)
                self.bn3 = nn.BatchNorm1d(embedding_dim)
        # if self.jk:
        #     self.jk_layer = JumpingKnowledge('lstm')
        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_dim + embedding_dim,
                                       embedding_dim)
        else:
            self.lin = None

    def _activation(self, name = 'relu'):
        assert name in ['relu', 'elu', 'leakyrelu']
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'elu':
            return nn.ELU(inplace=True)
        elif name =='leakyrelu':
            return nn.LeakyReLU(inplace=True)

    def _gcn(self,name, input_dim, hidden_dim,bias, activation='relu'):
        if name == 'SAGE':
            return DenseSAGEConv(input_dim, hidden_dim, normalize= True, bias= bias)
        else:
            nn1 = nn.Sequential(nn.Linear(input_dim,  hidden_dim), self._activation(activation),
                                nn.Linear( hidden_dim,  hidden_dim))
            return DenseGINConv(nn1)

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        # batch_size, num_nodes, in_channels = x.size()l


        x0 = x
        x1 = self.bn(1,self.active1(self.gcn1(x0, adj, mask, self.add_loop)))
        x2 = self.bn(2,self.active2(self.gcn2(x1, adj, mask, self.add_loop)))
        x3 = self.bn(3,self.active3(self.gcn3(x2, adj, mask, self.add_loop)))
        # if not self.jk:
        x = torch.cat([x1, x2, x3], dim=-1) #batch , node, (feat-dim * 3)
        if mask is not None:
            x = x * mask
        if self.lin is not None:
            x = self.lin(x)
            if mask is not None:
                x = x * mask
        return x

class CGC_Net(nn.Module):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, bias, bn, assign_hidden_dim,n_classes,
                 assign_ratio=0.25,  pred_hidden_dims=[50], concat = True, gcn_name='SAGE',
                 collect_assign = False, load_data_sparse = False,norm_adj=False,
                 activation = 'relu',  drop_out = 0.,jk = False):


        super(CGC_Net, self).__init__()

        self.jk = jk
        self.drop_out = drop_out
        self.norm_adj = norm_adj
        self.load_data_sparse = load_data_sparse
        self.collect_assign = collect_assign
        self.assign_matrix = []
        assign_dim = int(max_num_nodes * assign_ratio)
        self.GCN_embed_1 = GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn,
                                      add_loop= False, lin=False, gcn_name=gcn_name,activation=activation, jk = jk)
        if jk:
            self.jk1 = DenseJK('lstm',  hidden_dim, 3)
        self.GCN_pool_1 = GNN_Module(input_dim, assign_hidden_dim, assign_dim, bias, bn,
                                     add_loop= False, gcn_name=gcn_name,activation=activation, jk = jk)

        if concat and not jk:
            input_dim = hidden_dim * 2 + embedding_dim
        else:
            input_dim = embedding_dim

        assign_dim = int(assign_dim * assign_ratio)
        self.GCN_embed_2 = GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn,
                                      add_loop= False,lin=False, gcn_name=gcn_name,activation=activation, jk = jk)
        if jk:
            self.jk2 = DenseJK('lstm', hidden_dim , 3)
        self.GCN_pool_2 = GNN_Module(input_dim, assign_hidden_dim, assign_dim, bias, bn,
                                     add_loop= False, gcn_name=gcn_name,activation=activation, jk = jk)

        self.GCN_embed_3 = GNN_Module(input_dim, hidden_dim, embedding_dim, bias, bn,
                                      add_loop= False,lin=False, gcn_name=gcn_name,activation=activation, jk = jk)
        if jk:
            self.jk3 = DenseJK('lstm', hidden_dim, 3)
        pred_input = input_dim * 3
        self.pred_model = self.build_readout_module(pred_input, pred_hidden_dims,
                                                    n_classes, activation)


    @staticmethod
    def construct_mask( max_nodes, batch_num_nodes):
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()


    def _re_norm_adj(self,adj,p, mask = None):
        # pdb.set_trace()
        idx = torch.arange(0, adj.shape[1],out=torch.LongTensor())
        adj[:,idx,idx] = 0
        new_adj =  torch.div(adj,adj.sum(-1)[...,None] + EPS)*(1-p)
        new_adj[:,idx,idx] = p
        if mask is not None:
            new_adj = new_adj * mask
        return new_adj


    def _diff_pool(self, x, adj, s, mask):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s
        batch_size, num_nodes, _ = x.size()
        # [batch_size x num_nodes x next_lvl_num_nodes]
        s = torch.softmax(s, dim=-1)
        if self.collect_assign:
            self.assign_matrix.append(s.detach())
        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            s = s * mask
        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        return  out, out_adj


    def _activation(self, name = 'relu'):
        assert name in ['relu', 'elu', 'leakyrelu']
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'elu':
            return nn.ELU(inplace=True)
        elif name =='leakyrelu':
            return nn.LeakyReLU(inplace=True)

    def build_readout_module(self,pred_input_dim, pred_hidden_dims, n_classes, activation ):
        pred_input_dim = pred_input_dim
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, n_classes)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self._activation(activation))
                pred_input_dim = pred_dim
                if self.drop_out>0:
                    pred_layers.append(nn.Dropout(self.drop_out))
            pred_layers.append(nn.Linear(pred_dim, n_classes))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model


    def _sparse_to_dense_input(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        label = data.y
        edge_index = to_dense_adj(edge_index, batch)
        x ,batch_num_node= to_dense_batch(x, batch)
        return x, edge_index,batch_num_node,label

    def forward(self,  data):

        out_all = []
        mean_all = []
        self.assign_matrix = []
        if self.load_data_sparse:
            x, adj, batch_num_nodes, label = self._sparse_to_dense_input(data)
        else:
            x, adj, batch_num_nodes = data.x, data.edge_index, data.batch
        max_num_nodes = adj.size()[1]
        embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4, embedding_mask)
        embed_feature = self.GCN_embed_1(x, adj, embedding_mask)
        if self.jk:
            embed_feature = self.jk1(embed_feature)
        out, _ = torch.max(embed_feature, dim = 1)
        out_all.append(out)

        assign = self.GCN_pool_1(x, adj, embedding_mask)
        x, adj = self._diff_pool(embed_feature, adj, assign, embedding_mask)
        # stage 2
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4)
        embed_feature = self.GCN_embed_2(x, adj, None)
        if self.jk:
            embed_feature = self.jk2(embed_feature)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)
        assign = self.GCN_pool_2(x, adj, None)
        x, adj = self._diff_pool(embed_feature, adj, assign, None)
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4)
        embed_feature = self.GCN_embed_3(x, adj, None)
        if self.jk:
            embed_feature = self.jk3(embed_feature)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)
        output = torch.cat(out_all, 1)
        output = self.pred_model(output)
        # if self.training:
        #     cls_loss = F.cross_entropy(output, label, size_average=True)
        #     return output, cls_loss
        return output


class PlainGCN(torch.nn.Module):
    def __init__(self, n_classes=2):
        super(PlainGCN, self).__init__()
        self.conv1 = GraphConv(16, 128)
        self.pool1 = TopKPooling(128, ratio=0.7)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.7)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.7)
        self.lin1 = torch.nn.Linear(1152, 384)
        self.lin2 = torch.nn.Linear(384, 64)
        self.lin3 = torch.nn.Linear(64, n_classes)
        self.res_scale = 1

    def global_min_pool(self, x, batch, size=None):
        size = batch.max().item() + 1 if size is None else size
        return scatter_('min', x, batch, dim_size=size)

    def forward(self, data):
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        loc = x[:, 16:]
        x = x[:, :16]

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, perm1, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch), self.global_min_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))  # +x*self.res_scale
        x, edge_index, _, batch, perm2, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch), self.global_min_pool(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))  # +x*self.res_scale
        x, edge_index, _, batch, perm3, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch), self.global_min_pool(x, batch)], dim=1)

        x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x


class ResNet18(torch.nn.Module):
    '''
    CNN model feature
    '''

    def __init__(self, channel_size=3, num_classes=2):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(
            channel_size, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = torch.nn.Linear(512, num_classes)
        for m in self.resnet.conv1.modules():
            torch.nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu'
            )
        for m in self.resnet.fc.modules():
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def forward(self, img):
        out = self.resnet(img)
        return out


class ResNet34(torch.nn.Module):
    '''
    ClassifyNet feature
    '''
    def __init__(self, channel_size=3, num_classes=2):
        super().__init__()
        self.resnet = resnet34(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(
            channel_size, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = torch.nn.Linear(512, num_classes)
        for m in self.resnet.conv1.modules():
            torch.nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu'
            )
        for m in self.resnet.fc.modules():
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def forward(self, x):
        x = self.resnet(x)
        return x


class ResNet50(torch.nn.Module):
    '''
    ClassifyNet feature
    '''
    def __init__(self, channel_size=3, num_classes=2):
        super().__init__()
        self.resnet = resnet50(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(
            channel_size, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.resnet.fc = torch.nn.Linear(2048, num_classes)
        for m in self.resnet.conv1.modules():
            torch.nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu'
            )
        for m in self.resnet.fc.modules():
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def forward(self, x):
        x = self.resnet(x)
        return x


class MobileNetV2(torch.nn.Module):
    '''
    ClassifyNet feature
    '''
    def __init__(self, channel_size=3, num_classes=2):
        super().__init__()
        self.mobilenet = mobilenet_v2(pretrained=False)
        self.mobilenet.classifier[-1] = torch.nn.Linear(1280, num_classes)
        for m in self.mobilenet.classifier[-1].modules():
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def forward(self, x):
        x = self.mobilenet(x)
        return x


class DenseNet121(torch.nn.Module):
    '''
    ClassifyNet
    '''
    def __init__(self, channel_size=3, num_classes=2):
        super().__init__()
        self.densenet = densenet121(pretrained=False)
        self.densenet.features[0] = torch.nn.Conv2d(
            channel_size, 64, kernel_size=7, stride=2,
            padding=3, bias=False
        )
        self.densenet.classifier = torch.nn.Linear(1024, num_classes)
        for m in self.densenet.features[0].modules():
            torch.nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu'
            )
        for m in self.densenet.classifier.modules():
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def forward(self, x):
        x = self.densenet(x)
        return x


class EfficientNet(torch.nn.Module):
    '''
    EfficientNet
    '''
    def __init__(self, b, channel_size=3, num_classes=2):
        super().__init__()
        self.net = efficientnet.EfficientNet.from_pretrained(
            'efficientnet-b{}'.format(b)
        )
        # self.net._conv_stem = efficientnet.utils.Conv2dStaticSamePadding(
        #     channel_size, self.net._conv_stem.out_channels,
        #     kernel_size=(3, 3), stride=(2, 2), image_size=512
        # )
        self.net._fc = torch.nn.Linear(
            in_features=self.net._fc.in_features, out_features=num_classes,
            bias=True
        )
        # Initialize(self.net._conv_stem)
        # Initialize(self.net._fc)
        for m in self.net._fc.modules():
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def forward(self, x):
        return self.net(x)


class DeepGCN_Base(torch.nn.Module):
    """
    static graph
    """
    def __init__(self, block='res', in_channels=16, n_blocks=28, n_filters=256, n_heads=1,
                 conv='mr', act='relu', norm='batch', bias=True, dropout=0.2, n_classes=2):
        super(DeepGCN_Base, self).__init__()
        channels = n_filters
        heads = n_heads
        c_growth = 0
        self.n_blocks = n_blocks
        self.head = GraphConv(in_channels, channels, conv, act, norm, bias, heads)

        res_scale = 1 if block.lower() == 'res' else 0
        if block.lower() == 'dense':
            c_growth = channels
            self.backbone = MultiSeq(*[DenseGraphBlock(channels + i * c_growth, c_growth, conv, act, norm, bias, heads)
                                       for i in range(self.n_blocks - 1)])
        else:
            self.backbone = MultiSeq(*[ResGraphBlock(channels, conv, act, norm, bias, heads, res_scale)
                                       for _ in range(self.n_blocks - 1)])
        fusion_dims = int(channels * self.n_blocks + c_growth * ((1 + self.n_blocks - 1) * (self.n_blocks - 1) / 2))
        self.fusion_block = MLP([fusion_dims, 1024], act, norm, bias)
        self.pool = TopKPooling(1024, ratio=0.7)
        self.prediction = Seq(*[MLP([3072, 512], act, norm, bias), torch.nn.Dropout(p=dropout),
                                MLP([512, 256], act, norm, bias), torch.nn.Dropout(p=dropout),
                                MLP([256, n_classes], None, None, bias)])
        self.model_init()

    def global_min_pool(self, x, batch, size=None):
        size = batch.max().item() + 1 if size is None else size
        #         return scatter_('min', x, batch, dim=0, dim_size=size)
        return scatter_('min', x, batch, dim_size=size)

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, data):
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y - 1
        x = x[:, :16]
        feats = [self.head(x, edge_index)]
        for i in range(self.n_blocks - 1):
            feats.append(self.backbone[i](feats[-1], edge_index)[0])
        feats = torch.cat(feats, 1)
        fusion = self.fusion_block(feats)
        fusion, edge_index, _, batch, perm, _ = self.pool(fusion, edge_index, None, batch)
        fusion = torch.cat([gmp(fusion, batch), gap(fusion, batch), self.global_min_pool(fusion, batch)], dim=1)
        out = self.prediction(fusion)
        return out


def ResGCN14(**kwargs):
    model = DeepGCN_Base(
        block='res', in_channels=16, n_blocks=14, n_filters=256, n_heads=1,
        conv='mr', act='relu', norm='batch', bias=True, dropout=0.2, **kwargs)
    return model


def ResGCN28(**kwargs):
    model = DeepGCN_Base(
        block='res', in_channels=16, n_blocks=28, n_filters=256, n_heads=1,
        conv='mr', act='relu', norm='batch', bias=True, dropout=0.2, **kwargs)
    return model


def ResGCN56(**kwargs):
    model = DeepGCN_Base(
        block='res', in_channels=16, n_blocks=56, n_filters=256, n_heads=1,
        conv='mr', act='relu', norm='batch', bias=True, dropout=0.2, **kwargs)
    return model


def DenseGCN14(**kwargs):
    model = DeepGCN_Base(
        block='dense', in_channels=16, n_blocks=14, n_filters=256, n_heads=1,
        conv='mr', act='relu', norm='batch', bias=True, dropout=0.2, **kwargs)
    return model


def DenseGCN28(**kwargs):
    model = DeepGCN_Base(
        block='dense', in_channels=16, n_blocks=28, n_filters=256, n_heads=1,
        conv='mr', act='relu', norm='batch', bias=True, dropout=0.2, **kwargs)
    return model


def DenseGCN56(**kwargs):
    model = DeepGCN_Base(
        block='dense', in_channels=16, n_blocks=56, n_filters=256, n_heads=1,
        conv='mr', act='relu', norm='batch', bias=True, dropout=0.2, **kwargs)
    return model


def EfficientNetB0(channel_size, num_classes):
    return EfficientNet(0, channel_size, num_classes)

def EfficientNetB1(channel_size, num_classes):
    return EfficientNet(1, channel_size, num_classes)

def EfficientNetB2(channel_size, num_classes):
    return EfficientNet(2, channel_size, num_classes)

def EfficientNetB3(channel_size, num_classes):
    return EfficientNet(3, channel_size, num_classes)

def EfficientNetB4(channel_size, num_classes):
    return EfficientNet(4, channel_size, num_classes)

def EfficientNetB5(channel_size, num_classes):
    return EfficientNet(5, channel_size, num_classes)

def EfficientNetB6(channel_size, num_classes):
    return EfficientNet(6, channel_size, num_classes)

def EfficientNetB7(channel_size, num_classes):
    return EfficientNet(7, channel_size, num_classes)


def CGCNet(**kwargs):
    model = CGC_Net(max_num_nodes=2081,
                    input_dim=18, hidden_dim=20, embedding_dim=20, bias=True, bn=True, assign_hidden_dim=20,
                    assign_ratio=0.1, pred_hidden_dims=[50], concat=True,
                    gcn_name='SAGE', collect_assign=False,
                    load_data_sparse=True,
                    norm_adj=False, activation='relu', drop_out=0.,
                    jk=False,
                    **kwargs)
    return model