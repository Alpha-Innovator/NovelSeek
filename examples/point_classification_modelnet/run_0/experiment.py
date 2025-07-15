import os
from tqdm import tqdm
import pickle
import argparse
import pathlib
import json
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from metrics import ConfusionMatrix
import data_transforms
import traceback

"""
Model
"""


class STN3d(nn.Module):
    def __init__(self, in_channels):
        super(STN3d, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 9)
        )
        self.iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).reshape(1, 9)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv_layers(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.linear_layers(x)
        iden = self.iden.repeat(batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, k * k)
        )
        self.k = k
        self.iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).reshape(1, self.k * self.k)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv_layers(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.linear_layers(x)
        iden = self.iden.repeat(batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, in_channels=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(in_channels)
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024)
        )
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = self.conv_layer1(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class Model(nn.Module):
    def __init__(self, in_channels=3, num_classes=40, scale=0.001):
        super().__init__()
        self.mat_diff_loss_scale = scale
        self.backbone = PointNetEncoder(global_feat=True, feature_transform=True, in_channels=in_channels)
        self.cls_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(p=0.4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, gts):
        x, trans, trans_feat = self.backbone(x)
        x = self.cls_head(x)
        x = F.log_softmax(x, dim=1)
        loss = F.nll_loss(x, gts)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss, x


"""
dataset and normalization
"""


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class ModelNetDataset(Dataset):
    def __init__(self, data_root, num_category, num_points, split='train'):
        self.root = data_root
        self.npoints = num_points
        self.uniform = True
        self.use_normals = True
        self.num_category = num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.data_path = os.path.join(data_root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.data_path = os.path.join(data_root,
                                          'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        print('Load processed data from %s...' % self.data_path)
        with open(self.data_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]
        return point_set, label[0]

def main(config):
    pathlib.Path(config.out_dir).mkdir(parents=True, exist_ok=True)
    datasets, dataloaders = {}, {}
    for split in ['train', 'test']:
        datasets[split] = ModelNetDataset(config.data_root, config.num_category, config.num_point, split)
        dataloaders[split] = DataLoader(datasets[split], batch_size=config.batch_size, shuffle=(split == 'train'),
                                        drop_last=(split == 'train'), num_workers=8)

    model = Model(in_channels=config.in_channels).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate,
        betas=(0.9, 0.999), eps=1e-8,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.7
    )
    train_losses = []
    print("Training model...")
    model.train()
    global_step = 0
    cur_epoch = 0
    best_oa = 0
    best_acc = 0

    start_time = time.time()
    for epoch in tqdm(range(config.max_epoch), desc='training'):
        model.train()
        cm = ConfusionMatrix(num_classes=len(datasets['train'].classes))
        for points, target in tqdm(dataloaders['train'], desc=f'epoch {cur_epoch}/{config.max_epoch}'):
            # data transforms
            points = points.data.numpy()
            points = data_transforms.random_point_dropout(points)
            points[:, :, 0:3] = data_transforms.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = data_transforms.shift_point_cloud(points[:, :, 0:3])
            points = torch.from_numpy(points).transpose(2, 1).contiguous()

            points, target = points.cuda(), target.long().cuda()

            loss, logits = model(points, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2)
            optimizer.step()
            model.zero_grad()

            logs = {"loss": loss.detach().item()}
            train_losses.append(loss.detach().item())
            cm.update(logits.argmax(dim=1), target)

        scheduler.step()
        end_time = time.time()
        training_time = end_time - start_time
        macc, overallacc, accs = cm.all_acc()
        print(f"iter: {global_step}/{config.max_epoch * len(dataloaders['train'])}, \
              train_macc: {macc}, train_oa: {overallacc}")

        if (cur_epoch % config.val_per_epoch == 0 and cur_epoch != 0) or cur_epoch == (config.max_epoch - 1):
            model.eval()
            cm = ConfusionMatrix(num_classes=datasets['test'].num_category)
            pbar = tqdm(enumerate(dataloaders['test']), total=dataloaders['test'].__len__())
            for idx, (points, target) in pbar:
                points, target = points.cuda(), target.long().cuda()
                points = points.transpose(2, 1).contiguous()
                loss, logits = model(points, target)
                cm.update(logits.argmax(dim=1), target)

            tp, count = cm.tp, cm.count
            macc, overallacc, accs = cm.cal_acc(tp, count)
            print(f"iter: {global_step}/{config.max_epoch * len(dataloaders['train'])}, \
            val_macc: {macc}, val_oa: {overallacc}")

            if overallacc > best_oa:
                best_oa = overallacc
                best_acc = macc
                best_epoch = cur_epoch
                torch.save(model.state_dict(), os.path.join(config.out_dir, 'best.pth'))
        cur_epoch += 1

        print(f"finish epoch {cur_epoch} training")

    final_info = {
        "modelnet" + str(config.num_category): {
            "means": {
                "best_oa": best_oa,
                "best_acc": best_acc,
                "epoch": best_epoch
            }
        }
    }
    with open(os.path.join(config.out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument("--in_channels", type=int, default=6)
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--num_category", type=int, choices=[10, 40], default=40)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--val_per_epoch", type=int, default=5)
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print("Original error in subprocess:", flush=True)
        traceback.print_exc(file=open(os.path.join(args.out_dir, "traceback.log"), "w"))
        raise

    