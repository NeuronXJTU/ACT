import torch
import torch.utils.data.dataloader as dataloader
from torch_geometric.nn import DataParallel
import argparse
import os
import sys
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import random
from tqdm import tqdm
from collections import OrderedDict
from timm.loss import LabelSmoothingCrossEntropy
import timm.optim.optim_factory as optim_factory
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

from data import prepare_train_val_loader, ImageData
from common.utils import Logger
from opt import OptInit
import architecture
import pyramid_vig

from setting import CrossValidSetting as DataSetting
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)


setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]

GCN_LIST = ['ResGCN14', 'DenseGCN14', 'ResGCN14_0', 'ResGCN14_1', 'DenseGCN14_0', 'DenseGCN14_1', 'PlainGCN', 'CGCNet']
CNN_LIST = ['ResNet18', 'ResNet18_0', 'ResNet18_1', 'ResNet34', 'ResNet50', 'DenseNet121', 'MobileNetV2',
            'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']
ViG_LIST = ['pvig_ti_224_gelu', 'pvig_s_224_gelu', 'pvig_m_224_gelu', 'pvig_b_224_gelu']


def parse_args():
    parser = argparse.ArgumentParser(description='Recursion')
    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--channel_size', type=int, default=3)
    parser.add_argument('--phase', type=str, help='train or test')
    parser.add_argument('--pattern', type=str, help='GCN or CNN+GCN')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--nuclei_model', type=str, default='dla34up_epoch_1999.pth')
    parser.add_argument('--resume_gcn', type=str, default='')
    parser.add_argument('--resume_cnn', type=str, default='')
    parser.add_argument('--resume_vig', type=str, default='')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--txt_dir', type=str, default='./txts/')
    parser.add_argument('--train_txts', type=str, default='./txts/test.txt')
    parser.add_argument('--eval_txts', type=str, default='./txts/test.txt')
    parser.add_argument('--test_txts', type=str, default='./txts/test.txt')
    parser.add_argument('--tt_txts', type=str)
    parser.add_argument('--wsi_dir', type=str)
    parser.add_argument('--patches_dir', type=str)
    parser.add_argument('--masks_dir', type=str)
    parser.add_argument('--feature_save_path', type=str)
    parser.add_argument('--distance_save_path', type=str)
    parser.add_argument('--coordinate_save_path', type=str)

    parser.add_argument('--result', type=str, default='result/')
    # gcn data
    parser.add_argument('--gcndata_dir', dest='gcndata_dir', default='proto/fix_fuse_cia_knn/')
    parser.add_argument('--load_data_list', action='store_true', default=True)
    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type', default='ca',
                        help='[c, ca, cal, cl] c: coor, a:appearance, l:soft-label')
    parser.add_argument('--sample-time', dest='sample_time', default=1)
    parser.add_argument('--visualization', action='store_const', const=True, default=False,
                        help='use assignment matrix for visualization')
    parser.add_argument('--method', dest='method', help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix', help='suffix added to the output filename')
    parser.add_argument('--input_feature_dim', dest='input_feature_dim', type=int,
                        help='the feature number for each node', default=8)
    parser.add_argument('--skip_train', action='store_const',
                        const=True, default=False, help='only do evaluation')
    parser.add_argument('--normalize', default=False, help='normalize the adj matrix or not')
    parser.add_argument('--name', default='')
    parser.add_argument('--gcn_name', default='SAGE')
    parser.add_argument('--active', dest='activation', default='relu')
    parser.add_argument('--dynamic_graph', dest='dynamic_graph', action='store_const', const=True, default=False, )
    parser.add_argument('--sampling_method', default='random', )
    parser.add_argument('--test_epoch', default=5, type=int)
    parser.add_argument('--sita', default=1., type=float)

    parser.add_argument('--norm_adj', action='store_const', const=True, default=False, )
    parser.add_argument('--readout', default='max', type=str)
    parser.add_argument('--task', default='colon', type=str)
    parser.add_argument('--n', dest='neighbour', default=8, type=int)
    parser.add_argument('--sample_ratio', default=0.5, type=float)
    parser.add_argument('--drop', dest='drop_out', default=0., type=float)
    parser.add_argument('--noise', dest='add_noise', action='store_const', const=True, default=False, )
    parser.add_argument('--valid_full', action='store_const', const=True, default=False, )
    parser.add_argument('--dist_g', dest='distance_prob_graph', action='store_const', const=True, default=False, )
    parser.add_argument('--jk', dest='jump_knowledge', action='store_const', const=True, default=False)
    parser.add_argument('--g', dest='graph_sampler', default='knn', type=str)
    parser.add_argument('--cv', dest='cross_val', default=1, type=int)

    # deepGCN
    parser.add_argument('--n_blocks', default=14, type=int, help='number of basic blocks')
    parser.add_argument('--block', default='res', type=str, help='graph backbone block type {res, dense, plain}')
    parser.add_argument('--world_size', default=1, type=int, help='number of gpus')

    parser.set_defaults(datadir='proto/fix_fuse_cia_knn',
                        logdir='proto/fix_fuse_cia_knn/log',
                        resultdir='proto/fix_fuse_cia_knn/log',
                        sample_time=1,
                        dataset='nuclei',
                        max_nodes=2000,  # no use
                        cuda='0',
                        feature='cl',
                        lr=0.001,
                        clip=2.0,
                        batch_size=2,
                        num_epochs=1000,
                        num_workers=2,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=2,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='soft-assign',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1,
                        input_feature_dim=8,
                        dynamic_graph=False,
                        test_epoch=5,
                        )

    parser.add_argument('--out_dir', type=str, default='.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--image_txt', type=str, default='')
    parser.add_argument('--patient_txts', type=str, default='.')
    parser.add_argument('--valid_patient_txts', type=str, default='.')
    # tifs_txts
    # parser.add_argument('--tifs_result', type=str, default='.')
    parser.add_argument('--is_tiffile', type=bool, default=False)

    parser.add_argument('--featdir', type=str, default='wu')

    # DML
    parser.add_argument('--num_net', default=3, type=int)
    parser.add_argument('--net1', default='ResGCN14', choices=GCN_LIST + CNN_LIST + ViG_LIST, type=str)
    parser.add_argument('--net2', default='DenseGCN14', choices=GCN_LIST + CNN_LIST + ViG_LIST, type=str)
    parser.add_argument('--net3', default='ResNet18', choices=GCN_LIST + CNN_LIST + ViG_LIST, type=str)
    parser.add_argument('--kld', type=float, default=0.28)
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--output_dir', default=None, help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None, help='path where to nohup.log')

    # optimizer
    parser.add_argument('--blr', default=2e-3, type=float, help='initial learning rate')
    parser.add_argument('--optim', default='AdamW', choices=['AdamW', 'Adam', 'SGD', 'RMSprop'], type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--nesterov', default=True, type=bool)
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--step', default=20, type=int, metavar='N',
                        help='stepsize to decay learning rate (>0 means this is enabled)')

    args = parser.parse_args()
    return args


def remove_module(state_dict):
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    return new_state_dict


def run_epoch(epoch, phase, models, optimizers, data_loader, im_loader, args, nets, records, criterion_CE,
              criterion_KLD, criterion_LS):
    if phase == 'train':
        for i in range(args.num_net):
            models[i].train()
    else:
        for i in range(args.num_net):
            models[i].eval()

    loss_all = [0] * args.num_net
    loss_kld = [0] * args.num_net
    correct = [0] * args.num_net
    r_num = [0] * args.num_net  # actual sample num (drop_last)
    label_database = [list() for i in range(0, args.num_net)]
    pred_database = [list() for i in range(0, args.num_net)]
    sum_len = len(data_loader)

    wfs = []
    for net in nets:
        wf = open(os.path.join(args.output_dir, net, 'wf', str(epoch) + '.txt'), 'w')
        wfs.append(wf)

    for index, (data, (image, image_path)) in enumerate(zip(data_loader, im_loader)):
        image = image.to(args.device)
        if args.num_classes == 7:  # LUNG_256_random
            labels = torch.cat([d.y - 1 for d in data]).to(args.device)
        elif args.num_classes == 3:  # CRC_new
            labels = torch.cat([d.y for d in data]).to(args.device)

        outputs = []
        losses = []
        CE_loss = []
        KLD_loss = []
        if phase == 'train':
            for i in range(args.num_net):
                if nets[i] in GCN_LIST:
                    outputs.append(models[i](data))
                else:
                    outputs.append(models[i](image))
            for k in range(args.num_net):
                if nets[i] in GCN_LIST:
                    CE_loss.append(criterion_CE(outputs[k], labels))
                else:
                    CE_loss.append(criterion_LS(outputs[k], labels))
                KLD_loss.append(0)
                for l in range(args.num_net):
                    if l != k:
                        KLD_loss[k] += criterion_KLD(F.log_softmax(outputs[k], dim=1),
                                                     F.softmax(outputs[l], dim=1).detach())
                if KLD_loss[k] < args.kld:
                    losses.append(CE_loss[k])
                else:
                    losses.append(CE_loss[k] + KLD_loss[k] / (args.num_net - 1))
            for i in range(args.num_net):
                optimizers[i].zero_grad()
                losses[i].backward()
                optimizers[i].step()
        else:
            with torch.no_grad():
                for i in range(args.num_net):
                    if nets[i] in GCN_LIST:
                        outputs.append(models[i](data))
                    else:
                        outputs.append(models[i](image))
                for k in range(args.num_net):
                    CE_loss.append(criterion_CE(outputs[k], labels))
                    KLD_loss.append(0)
                    for l in range(args.num_net):
                        if l != k:
                            KLD_loss[k] += criterion_KLD(F.log_softmax(outputs[k], dim=1),
                                                         F.softmax(outputs[l], dim=1).detach())
                    #                     losses.append(CE_loss[k] + KLD_loss[k] / (args.num_net - 1))
                    losses.append(CE_loss[k])

        labels = labels.cpu().data.numpy()
        preds = []
        for i in range(args.num_net):
            preds.append(outputs[i].max(dim=1)[1].cpu().data.numpy())

        image_path = np.array(image_path)
        for i in range(args.num_net):
            # record wrong images' path
            output_preds_sub = (preds[i] == labels)
            indexw = np.arange(0, len(preds[i]))
            indexw = indexw[output_preds_sub == False]
            wrong_image = image_path[indexw]
            for wim in wrong_image:
                wfs[i].write(wim + '\n')
            wfs[i].flush()

            r_num[i] += len(preds[i])
            right = np.sum(preds[i] == labels)
            loss_all[i] += losses[i].cpu().data.numpy()

            loss_kld[i] += KLD_loss[i]  # kld for num_net=1
            pred_database[i] += preds[i].tolist()
            label_database[i] += labels.tolist()
            correct[i] += right

            message = 'Model=[{}] Epoch=[{}/{}] iter=[{}/{}] {} loss=[{:.5f}] {} right=[{}/{}]'.format(
                nets[i], epoch, args.epochs, index + 1, sum_len, phase, losses[i].cpu().data.numpy(), phase, right,
                len(preds[i]))
            records[i].write(message + '\n')
            records[i].flush()
            print(message)

    print("ending...")
    loss_average = []
    acc = []
    class_reports = []
    for i in range(args.num_net):
        records[i].close()
        wfs[i].close()
        loss_average.append(loss_all[i] / len(data_loader))
        print("loss_kld[{}] = {}".format(i, loss_kld[i] / len(data_loader)))
        acc.append(correct[i] / r_num[i])
        if args.num_classes == 7:
            target_names = ['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5', 'Type 6', 'Type 12']
        elif args.num_classes == 3:
            target_names = ['Type 1', 'Type 2', 'Type 3']
        class_reports.append(
            metrics.classification_report(label_database[i], pred_database[i], target_names=target_names, digits=4))

    return loss_average, acc, class_reports


def train(args):
    if args.num_net == 3:
        nets = [args.net1, args.net2, args.net3]
    elif args.num_net == 2:
        nets = [args.net1, args.net2]
    elif args.num_net == 1:
        nets = [args.net1]
    args.num_net = len(nets)

    # tensorboard
    if not os.path.exists(os.path.join(args.log_dir, 'run')):
        os.makedirs(os.path.join(args.log_dir, 'run'))
    log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'run'))

    # models
    models = []
    optimizers = []
    schedulers = []
    for i in range(args.num_net):
        if nets[i] in GCN_LIST:
            model_name = nets[i].split('_')[0]
            model = architecture.__dict__[model_name](n_classes=args.num_classes)
            if args.resume_gcn != '':
                print('=> loading checkpoint {}'.format(args.resume_gcn))
                checkpoint = torch.load(args.resume_gcn)
                start_epoch = checkpoint['epoch'] + 1
                model.load_state_dict(remove_module(checkpoint['state_dict']))
                print('=> successfully loaded.')
            model = DataParallel(model).to(args.device)
            models.append(model)
        elif nets[i] in CNN_LIST:
            model_name = nets[i].split('_')[0]
            im_model = architecture.__dict__[model_name](channel_size=3, num_classes=args.num_classes)
            if args.resume_cnn != '':
                print('=> loading checkpoint {}'.format(args.resume_cnn))
                checkpoint = torch.load(args.resume_cnn)
                start_epoch = checkpoint['epoch'] + 1
                im_model.load_state_dict(remove_module(checkpoint['state_dict']))
                print('=> successfully loaded.')
            im_model = torch.nn.DataParallel(im_model).to(args.device)
            models.append(im_model)
        elif nets[i] in ViG_LIST:
            model_name = nets[i]
            im_model = pyramid_vig.__dict__[model_name](num_classes=args.num_classes)
            if args.resume_vig != '':
                print('=> loading checkpoint {}'.format(args.resume_vig))
                checkpoint = torch.load(args.resume_vig)
                start_epoch = checkpoint['epoch'] + 1
                im_model.load_state_dict(remove_module(checkpoint['state_dict']))
                print('=> successfully loaded.')
            im_model = torch.nn.DataParallel(im_model).to(args.device)
            models.append(im_model)
        else:
            print('Net type not supported! Supported type are: ', GCN_LIST, CNN_LIST, ViG_LIST)
            exit()

    # print models
    for i in range(args.num_net):
        print('Model {}: {}'.format(i, nets[i]))
        print(models[i])

    # output txt
    records = []
    for net in nets:
        if not os.path.exists(os.path.join(args.output_dir, net, 'pth')):
            os.makedirs(os.path.join(args.output_dir, net, 'pth'))
        if not os.path.exists(os.path.join(args.output_dir, net, 'train')):
            os.makedirs(os.path.join(args.output_dir, net, 'train'))
        if not os.path.exists(os.path.join(args.output_dir, net, 'test')):
            os.makedirs(os.path.join(args.output_dir, net, 'test'))
        if not os.path.exists(os.path.join(args.output_dir, net, 'wf')):
            os.makedirs(os.path.join(args.output_dir, net, 'wf'))
        record = open(os.path.join(args.output_dir, net, 'record.txt'), 'w')
        records.append(record)

    # graph dataset
    train_loader, _, test_loader = prepare_train_val_loader(args)
    train_loader.dataset.set_epoch(0)

    # image dataset
    train_set = ImageData('train', args)
    eval_set = ImageData('eval', args)
    print('len train_set', len(train_set), 'len eval_set', len(eval_set))
    imtrain_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    imeval_loader = torch.utils.data.DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )

    # lr
    actual_lr = args.blr * args.batch_size * args.world_size / 128

    # optimizer
    for i in range(args.num_net):
        if args.optim == 'SGD':
            optimizers.append(
                torch.optim.SGD(models[i].parameters(), lr=actual_lr, momentum=args.momentum, weight_decay=args.decay,
                                nesterov=args.nesterov))
        elif args.optim == 'Adam':
            optimizers.append(torch.optim.Adam(models[i].parameters(), lr=actual_lr, weight_decay=args.decay))
        elif args.optim == 'RMSprop':
            optimizers.append(
                torch.optim.RMSprop(models[i].parameters(), lr=actual_lr, momentum=args.momentum,
                                    weight_decay=args.decay))
        elif args.optim == 'AdamW':
            param_groups = optim_factory.add_weight_decay(models[i], args.decay)
            optimizers.append(torch.optim.AdamW(param_groups, lr=actual_lr, betas=(0.9, 0.95)))
        else:
            print("Choose your optimizer! choices=['AdamW', 'Adam', 'SGD', 'RMSprop']")
            exit(0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[i], patience=3, factor=0.5, verbose=True,
                                                               min_lr=1e-8)
        schedulers.append(scheduler)

    # loss
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_KLD = torch.nn.KLDivLoss(reduction='batchmean')
    criterion_LS = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    # start training
    best_acc = [0] * args.num_net
    for epoch in tqdm(range(0, args.epochs)):
        epoch_train_records = []
        epoch_test_records = []
        for net in nets:
            epoch_train_record = open(os.path.join(args.output_dir, net, 'train', str(epoch) + '.txt'), 'w')
            epoch_test_record = open(os.path.join(args.output_dir, net, 'test', str(epoch) + '.txt'), 'w')
            epoch_train_records.append(epoch_train_record)
            epoch_test_records.append(epoch_test_record)
        # train one epoch
        train_loss, train_acc, class_report_tr = run_epoch(epoch, 'train',
                                                           models, optimizers, train_loader, imtrain_loader,
                                                           args, nets, epoch_train_records,
                                                           criterion_CE, criterion_KLD, criterion_LS)
        # test one epoch
        test_loss, test_acc, class_report_te = run_epoch(epoch, 'test',
                                                         models, optimizers, test_loader, imeval_loader,
                                                         args, nets, epoch_test_records,
                                                         criterion_CE, criterion_KLD, criterion_LS)

        for i in range(args.num_net):
            schedulers[i].step(train_loss[i])

        for i in range(args.num_net):
            if best_acc[i] < test_acc[i]:
                best_acc[i] = test_acc[i]
                torch.save(
                    {
                        'epoch': epoch,
                        'acc': best_acc[i],
                        'lr': optimizers[i].param_groups[0]['lr'],
                        'state_dict': models[i].state_dict(),
                    }, os.path.join(
                        args.output_dir, nets[i], 'pth',
                        args.output_dir.split('/')[-1] + '_best.pth'
                    )
                )
            message = 'Epoch: {:03d}, Train Loss: {:.5f}, Train Acc: {:.5f}\n{}\n' \
                      'Test Loss: {:.5f}, Test Acc: {:.5f}, Best Acc: {:.5f}\n{}\n'.format(
                epoch, train_loss[i], train_acc[i], class_report_tr[i],
                test_loss[i], test_acc[i], best_acc[i], class_report_te[i])
            print(message)
            records[i].write(message)
            records[i].flush()

            # tensorboard
            if log_writer is not None:
                log_writer.add_scalar(nets[i] + '/train_acc', train_acc[i], epoch)
                log_writer.add_scalar(nets[i] + '/train_loss', train_loss[i], epoch)
                log_writer.add_scalar(nets[i] + '/test_acc', test_acc[i], epoch)
                log_writer.add_scalar(nets[i] + '/test_loss', test_loss[i], epoch)

            # save model
            if epoch % 50 == 0 or epoch == args.epochs - 1:
                torch.save(
                    {
                        'epoch': epoch,
                        'acc': test_acc[i],
                        'lr': optimizers[i].param_groups[0]['lr'],
                        'state_dict': models[i].state_dict(),
                    }, os.path.join(
                        args.output_dir, nets[i], 'pth',
                        args.output_dir.split('/')[-1] + '_epoch_' + str(epoch) + '.pth'
                    )
                )
    for i in range(args.num_net):
        records[i].close()


if __name__ == "__main__":
    args = parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.visualization = False
    args.is_tiffile = False
    args.pattern = 'DML'
    # args.resume_gcn = ''
    # args.resume_cnn = ''
    args.blr = 2e-3

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print('begin==>')

    train(args)
