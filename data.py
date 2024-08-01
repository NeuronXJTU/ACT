import os.path as osp
import os
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data import DataListLoader
from torch_geometric.nn import radius_graph
import random
import numpy as np
from tqdm import tqdm
from common.utils import FarthestSampler, filter_sampled_indice
# from torch_geometric.utils.sparse import sparse_to_dense
from torch_geometric.utils import to_dense_adj
from setting import CrossValidSetting
import cv2
import albumentations as albu
import albumentations.augmentations.functional as F

_MEAN_CIA = {1: [1.44855589e+02, 1.50849152e+01, 4.16993829e+02, -9.89115031e-02,
                 4.29073361e+00, 7.03308534e+00, 1.50311764e-01, 1.20372119e-01,
                 1.99874447e-02, 7.24825770e-01, 1.28062193e+02, 1.71914904e+01,
                 9.00313323e+00, 4.29522533e+01, 8.76540101e-01, 8.06801284e+01, 3584, 3584],
             2: [1.45949547e+02, 1.53704952e+01, 4.39127922e+02, -1.10080479e-01,
                 4.30617772e+00, 7.27624697e+00, 1.45825849e-01, 1.21214980e-01,
                 2.03645262e-02, 7.28225987e-01, 1.27914898e+02, 1.72524907e+01,
                 8.96012595e+00, 4.30067152e+01, 8.76016742e-01, 8.09466370e+01, 3584, 3584],
             3: [1.45649518e+02, 1.52438912e+01, 4.30302592e+02, -1.07054163e-01,
                 4.29877990e+00, 7.13800092e+00, 1.47971754e-01, 1.20517868e-01,
                 2.00830612e-02, 7.24701226e-01, 1.26430193e+02, 1.71710396e+01,
                 8.94070628e+00, 4.27421136e+01, 8.74665450e-01, 8.02611304e+01, 3584, 3584]}

_STD_CIA = {1: [3.83891570e+01, 1.23159786e+01, 3.74384781e+02, 5.05079918e-01,
                1.91811771e-01, 2.95460595e+00, 7.31040425e-02, 7.41484835e-02,
                2.84762625e-02, 2.47544275e-01, 1.51846534e+02, 5.96200235e+01,
                6.00087195e+00, 2.85961395e+01, 1.95532620e-01, 5.49411936e+01, 3584, 3584],
            2: [3.86514982e+01, 1.25207234e+01, 3.87362858e+02, 5.02515226e-01,
                1.89045551e-01, 3.05856764e+00, 7.22404102e-02, 7.53090608e-02,
                2.90460236e-02, 2.46734916e-01, 1.53743958e+02, 6.34661492e+01,
                6.02575043e+00, 2.88403590e+01, 1.94214810e-01, 5.49984596e+01, 3584, 3584],
            3: [3.72861596e+01, 1.23840868e+01, 3.87834784e+02, 5.02444847e-01,
                1.86722327e-01, 2.99248449e+00, 7.20327363e-02, 7.45553798e-02,
                2.87285660e-02, 2.49195190e-01, 1.50986869e+02, 6.56370060e+01,
                6.00008814e+00, 2.86376250e+01, 1.97764021e-01, 5.54134874e+01, 3584, 3584]}


def prepare_train_val_loader(args):
    setting = CrossValidSetting()
    sampler_type = None
    train_dataset_loader = DataListLoader(NucleiDatasetBatchOutput(
        root=setting.root,
        feature_type=args.feature_type,
        split='train', sampling_time=setting.sample_time,
        sampling_ratio=args.sample_ratio,
        normalize=args.normalize, dynamic_graph=args.dynamic_graph,
        sampling_method=args.sampling_method,
        datasetting=setting,
        neighbour=args.neighbour,
        graph_sampler=args.graph_sampler,
        crossval=args.cross_val, filepath=args.patient_txts, is_tiffile=args.is_tiffile),
        sampler=sampler_type,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    validset = NucleiDatasetBatchOutput(root=setting.root,
                                        feature_type=args.feature_type,
                                        split='valid',
                                        sampling_time=setting.sample_time,
                                        sampling_ratio=args.sample_ratio,
                                        normalize=args.normalize,
                                        dynamic_graph=args.dynamic_graph,
                                        sampling_method=args.sampling_method,
                                        datasetting=setting,
                                        neighbour=args.neighbour,
                                        graph_sampler=args.graph_sampler,
                                        crossval=args.cross_val,
                                        filepath=args.valid_patient_txts,
                                        is_tiffile=args.is_tiffile)

    val_dataset_loader = DataListLoader(
        validset,
#         batch_size=setting.batch_size,  # setting.batch_size
        batch_size=args.batch_size,  # setting.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    if not args.visualization:
        test_dataset_loader = val_dataset_loader
    else:
        testset = NucleiDatasetTest(root=setting.root,
                                    feature_type=args.feature_type,
                                    split='valid',
                                    normalize=args.normalize,
                                    sampling_method=args.sampling_method,
                                    datasetting=setting,
                                    neighbour=args.neighbour,
                                    graph_sampler=args.graph_sampler,
                                    crossval=args.cross_val)

        test_dataset_loader = torch.utils.data.DataLoader(
            testset,
#             batch_size=setting.batch_size,
            batch_size=args.batch_size,  # setting.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    return train_dataset_loader, val_dataset_loader, test_dataset_loader


def prepare_predict_loader(args):
    setting = CrossValidSetting()
    # this is for visualization test set
    testset = NucleiDatasetBatchOutput(root=setting.root,
                                       feature_type=args.feature_type,
                                       split='predict',
                                       sampling_time=setting.sample_time,
                                       sampling_ratio=args.sample_ratio,
                                       normalize=args.normalize,
                                       dynamic_graph=args.dynamic_graph,
                                       sampling_method=args.sampling_method,
                                       datasetting=setting,
                                       neighbour=args.neighbour,
                                       graph_sampler=args.graph_sampler, crossval=args.cross_val,
                                       filepath=args.patient_txts, is_tiffile=args.is_tiffile)
    test_dataset_loader = DataListLoader(
        testset,
#         batch_size=setting.batch_size,  # setting.batch_size,
        batch_size=args.batch_size,  # setting.batch_size,
        shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    return test_dataset_loader


class NucleiDataset(Dataset):

    def __init__(self, root, feature_type, transform=None, pre_transform=None, split='train',
                 sampling_time=10, sampling_ratio=0.5, normalize=False, dynamic_graph=False, sampling_method='farthest',
                 datasetting=None, neighbour=8, graph_sampler='knn', crossval=1, filepath='patient_100_node_0',
                 is_tiffile=False):
        super(NucleiDataset, self).__init__(root, transform, pre_transform)
        setting = datasetting  # DataSetting()
        self.epoch = 0
        self.val_epoch = 0

        self.graph_sampler = graph_sampler
        self.task = setting.name
        self.setting = setting
        self.is_tiffile = is_tiffile
        self.dynamic_graph = dynamic_graph
        self.sample_method = sampling_method
        self.sampling_by_ratio = True
        if self.dynamic_graph and sampling_method != 'random':
            self.sampler = FarthestSampler()
        self.normalize = normalize
        self.sampling_time = sampling_time
        self.sampling_ratio = sampling_ratio
        self.max_edge_distance = setting.max_edge_distance
        self.max_num_nodes = int(setting.max_num_nodes * sampling_ratio) + 1
        self.max_neighbours = neighbour
        self.cross_val = crossval
        self.split = split
        if self.split != 'train':
            np.random.seed(1024)
        else:
            np.random.seed(None)

        self.processed_root = os.path.join(self.root)

        self.original_files = []
        self.original_files = [f.split('.npy')[0] for f in self.original_files]
        self.mean = _MEAN_CIA[crossval]
        self.std = _STD_CIA[crossval]
        if feature_type == 'c':
            self.mean = self.mean[-2:]
            self.std = self.std[-2:]
        elif feature_type == 'a':
            self.mean = self.mean[:-2]
            self.std = self.std[:-2]
        self.mean = torch.Tensor(self.mean)
        self.std = torch.Tensor(self.std)
        self.idxlist = []
        # folder is file name
        """ for folder in self.processed_dir:
            f1 = open(folder,'r')
            for line in f1.readlines():
                self.idxlist.append(line.strip()) """
        f1 = open(filepath, 'r')
        for line in f1.readlines():
            self.idxlist.append(line.strip())
            # self.idxlist.extend([os.path.join(folder.split('/')[-1],f) for f in os.listdir(folder)])

        '''
        for folder in self.processed_dir:
            #print('folder',folder)
            self.idxlist.extend([os.path.join(folder.split('/')[-1],f) for f in os.listdir(folder)])
        '''
        # print('self.idxlist', self.idxlist)
        print('len self.idxlist2 ', len(self.idxlist))
        # input('w?')
        self.feature_type = feature_type

    @property
    def raw_file_names(self):
        return self.original_files

    @property
    def raw_paths(self):
        pass

    @property
    def processed_file_names(self):
        processed_file_names = self.idxlist
        return processed_file_names

    def len(self):
        return len(self.processed_file_names)

    def _download(self):
        # Download to `self.raw_dir`.
        pass

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_val_epoch(self, epoch):
        self.val_epoch = epoch

    def _process(self):
        pass

    def _sampling(self, num_sample, ratio, distance_path=None):

        if self.sampling_by_ratio:
            num_subsample = int(num_sample * ratio)
            if self.task != 'colon':
                if num_sample < 100:
                    num_subsample = num_sample

        else:
            num_subsample = min(self.max_num_nodes, num_sample)
        if self.sample_method == 'farthest':

            distance = np.load(distance_path.replace('.pt', '.npy'))
            indice = self.sampler(distance, num_subsample)

        elif self.sample_method == 'fuse':
            # 70% farthest, 30% random
            far_num = int(0.7 * num_subsample)
            rand_num = num_subsample - far_num
            distance = np.load(distance_path.replace('.pt', '.npy'))
            far_indice = self.sampler(distance, far_num)
            remain_item = filter_sampled_indice(far_indice, num_sample)
            rand_indice = np.asarray(random.sample(remain_item, rand_num))

            indice = np.concatenate((far_indice, rand_indice), 0)

        else:
            # random
            indice = np.random.choice(num_sample, num_subsample, replace=False)

        return indice, num_subsample

    def process(self):
        pass

    def get(self, idx):

        adj_all = torch.zeros((self.max_num_nodes, self.max_num_nodes), dtype=torch.float)
        # transform
        if self.dynamic_graph:
            data = torch.load(osp.join('zhengtu', self.idxlist[idx]))
            num_nodes = data.num_nodes
            dist_path = os.path.join(self.root, 'distance', self.setting.dataset, self.idxlist[idx])
            choice, sample_num_node = self._sampling(num_nodes, self.sampling_ratio, dist_path)
            for key, item in data:
                if torch.is_tensor(item) and item.size(0) == num_nodes:
                    data[key] = item[choice]

            # generate the graph
            if self.graph_sampler == 'knn':
                edge_index = radius_graph(data.x, self.max_edge_distance, None, True, self.max_neighbours)
#                 edge_index = radius_graph(data.pos, self.max_edge_distance, None, True, self.max_neighbours) # ablation_cell_graph
                adj = to_dense_adj(edge_index)
            else:
                raise NotImplementedError
            adj[adj > 0] = 1
            adj_all[: sample_num_node, :sample_num_node] = adj
        else:
            data = torch.load(osp.join(str(self.epoch), self.idxlist[idx]))
            if self.graph_sampler == 'knn':
                edge_index = radius_graph(data.x, self.max_edge_distance, None, True, self.max_neighbours, num_workers=4)
#                 edge_index = radius_graph(data.pos, self.max_edge_distance, None, True, self.max_neighbours, num_workers=4) # ablation_cell_graph
                adj = to_dense_adj(edge_index)
            else:
                raise NotImplementedError
            num_nodes = adj.shape[0]
            adj_all[: num_nodes, :num_nodes] = adj
        feature = data.x
        if self.feature_type == 'c':
            feature = feature[:, -2:]
        elif self.feature_type == 'a':
            feature = feature[:, :-2]
        num_feature = feature.shape[1]
        feature = (feature - self.mean) / self.std
        feature_all = torch.zeros((self.max_num_nodes, num_feature), dtype=torch.float)
        if self.dynamic_graph:
            feature_all[:sample_num_node] = feature
        else:
            feature_all[:num_nodes] = feature
        label = data.y
        idx = torch.tensor(idx)
        return {'adj': adj_all,
                'feats': feature_all,
                'label': label,
                'num_nodes': sample_num_node if self.dynamic_graph else num_nodes,
                'patch_idx': idx}


class NucleiDatasetTest(NucleiDataset):
    def __init__(self, root, feature_type, transform=None, pre_transform=None, split='train',
                 normalize=False,
                 sampling_method='farthest', datasetting=None,
                 neighbour=8, graph_sampler='knn', crossval=1, filepath='patient_100_node_4.txt', is_tiffile=False):
        super(NucleiDatasetTest, self).__init__(root, feature_type, transform=transform,
                                                pre_transform=pre_transform, split=split,
                                                sampling_time=1, sampling_ratio=1,
                                                normalize=normalize, dynamic_graph=False,
                                                sampling_method=sampling_method,
                                                datasetting=datasetting, neighbour=neighbour,
                                                graph_sampler=graph_sampler, crossval=crossval, filepath=filepath,
                                                is_tiffile=is_tiffile)

    def get(self, idx):
        # only support batch size = 1
        data = torch.load(osp.join('zhengtu', self.idxlist[idx]))
        # generate the graph
        if self.graph_sampler == 'knn':
            edge_index = radius_graph(data.x, self.max_edge_distance, None, True, self.max_neighbours, num_workers=4)
#             edge_index = radius_graph(data.pos, self.max_edge_distance, None, True, self.max_neighbours, num_workers=4) # ablation_cell_graph
            adj_all = to_dense_adj(edge_index)
        else:
            raise NotImplementedError
        adj_all[adj_all > 0] = 1
        num_nodes = adj_all.shape[0]
        feature = data.x
        if self.feature_type == 'c':
            feature = feature[:, -2:]
        elif self.feature_type == 'a':
            feature = feature[:, :-2]
        if not is_tiffile:
            feature = (feature - self.mean) / self.std
        label = data.y
        idx = torch.tensor([idx])
        return {'adj': adj_all,
                'feats': feature,
                'label': label,
                'num_nodes': num_nodes if self.dynamic_graph else num_nodes,
                'patch_idx': idx
                }


def get_augment(min_area=0., min_visibility=0.):
    # aug = [albu.Resize(height=size, width=size, p=1)]
    aug = [
        # albu.Resize(height=224, width=224, p=1),
        # albu.ShiftScaleRotate(
        #     shift_limit=cfg.shift_limit, scale_limit=0.0,
        #     rotate_limit=0.0, interpolation=1, p=0.5
        # ),
        # albu.RandomScale(scale_limit=cfg.scale_limit, p=1),
        #         albu.PadIfNeeded(
        #             min_height=512, min_width=512
        #         ),
        # albu.RandomCrop(height=cfg.input_size, width=cfg.input_size, p=1)
        # albu.CenterCrop(height=512, width=512, p=1)
    ]
    aug += [
        albu.OneOf(
            [
                albu.NoOp(),
                albu.VerticalFlip(p=1),
                albu.HorizontalFlip(p=1),
                albu.RandomRotate90(p=1),
                albu.Transpose(p=1),
            ]
        ),
        # albu.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5),
        # albu.RandomBrightnessContrast(
        #    p=0.5, brightness_limit=0.1, contrast_limit=0.1
        # ),
        # albu.OneOf(
        #    [
        #        albu.GaussianBlur(blur_limit=3, p=1),
        #        albu.MedianBlur(blur_limit=3, p=1),
        #        albu.MotionBlur(blur_limit=3, p=1),
        #    ], p=0.5
        # )
    ]
    return albu.Compose(aug)


class NucleiDatasetBatchOutput(NucleiDataset):
    def __init__(self, root, feature_type, transform=None, pre_transform=None, split='train',
                 sampling_time=10, sampling_ratio=0.5, normalize=False, dynamic_graph=False, sampling_method='farthest',
                 datasetting=None, neighbour=8, graph_sampler='knn', crossval=1, filepath='filepath', is_tiffile=False):
        super(NucleiDatasetBatchOutput, self).__init__(root, feature_type, transform=transform,
                                                       pre_transform=pre_transform, split=split,
                                                       sampling_time=sampling_time, sampling_ratio=sampling_ratio,
                                                       normalize=normalize, dynamic_graph=dynamic_graph,
                                                       sampling_method=sampling_method
                                                       , datasetting=datasetting, neighbour=neighbour,
                                                       graph_sampler=graph_sampler, crossval=crossval,
                                                       filepath=filepath, is_tiffile=False)

        self.aug = get_augment()

    def get(self, idx):
        # print('q    ?',idx)

        epoch = self.epoch if self.split == 'train' else self.val_epoch
        if self.dynamic_graph:
            data = torch.load(self.idxlist[idx])
        else:
            data = torch.load(self.idxlist[idx])
        #         print('data3 ',data)
        if self.feature_type == 'c':
            data.x = data.x[:, -2:]
        elif self.feature_type == 'a':
            data.x = data.x[:, :-2]
        if self.dynamic_graph:
            num_nodes = data.num_nodes
            dist_path = os.path.join('/media/sensetime/7cfef7ad-62af-4069-a59e-dc2c861ef336/CGC_Net/gcn_cam/new_proto',
                                     'distance', self.setting.dataset, self.idxlist[idx])
            choice, _ = self._sampling(num_nodes, self.sampling_ratio, dist_path)
            for key, item in data:
                if torch.is_tensor(item) and item.size(0) == num_nodes:
                    data[key] = item[choice]
        if self.graph_sampler == 'knn':
            edge_index = radius_graph(data.x, self.max_edge_distance, None, True, self.max_neighbours, num_workers=4)
#             edge_index = radius_graph(data.pos, self.max_edge_distance, None, True, self.max_neighbours, num_workers=4) # ablation_cell_graph
            data.edge_index = edge_index
        else:
            raise NotImplementedError
        data.patch_idx = torch.tensor([idx])
        # print(data.x.shape,'   ',self.mean.shape)
        if self.is_tiffile:
            data.x = (data.x - self.mean) / self.std
        return data


class ImageData(torch.utils.data.Dataset):
    '''
    Data
    '''

    # phase in {'train', 'eval'}
    def __init__(self, phase, args):
        super().__init__()
        # basepath = '/mnt/lustre/xudou/bisai/CGC_Net/CGC-Net-master/cam17/raw/CELL/fold_1/'
        # basepath = '/media/sensetime/7cfef7ad-62af-4069-a59e-dc2c861ef336/CGC_Net/patchs/pos_negpatches/'
        basepath = ''  # '/mnt/lustre/xudou/cam17_cls/gcn_cam/patches/fold_1'
        self.phase = phase
        self.args = args
        self.images = {}
        # for line in open('txts/camelyon17_small.txt').readlines():
        # for line in open('txts/camelyon17.txt').readlines():
        # for line in open('txts/CoNSep_patch.txt').readlines():
        for line in open(args.image_txt).readlines():
            name = line.split('/')[-1].strip()
            line = line.strip()
            self.images[name] = line
        if phase == 'train':
            files = [args.patient_txts]
        elif phase == 'eval':
            files = [args.valid_patient_txts]
        image_names = []
        for file_name in files:
            with open(file_name) as reader:
                lines = reader.readlines()
            for line in lines:
                line = line.strip()
                image_names.append(self.images[line.strip().split('/')[-1].split('.pt')[0]])
        self.image_names = image_names
        self.aug = get_augment()

    def __getitem__(self, index):
        kind = 4
        mode = random.randint(0, 2 * kind - 1)
        # print(self.image_names[index])
        image = cv2.imread(self.image_names[index])
        image = cv2.resize(image, (224, 224))
        # print(self.image_names[index])
        image = {'image': image, }
        image = self.aug(**image)
        image = image['image']
        image = np.transpose(image, (2, 0, 1)).copy()
        return torch.from_numpy(image).float().div(255), self.image_names[index]

    def __len__(self):
        return len(self.image_names)
