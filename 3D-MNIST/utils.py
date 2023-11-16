import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
from networks import Conv3DNet

import tqdm

import random
import h5py

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_and_print(dirname, msg):
    if not os.path.isfile(dirname):
        f = open(dirname, "w")
        f.write(str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())))
        f.write("\n")
        f.close()
    f = open(dirname, "a")
    f.write(str(msg))
    f.write("\n")
    f.close()
    print(msg)

def get_images(images_all, indices_class, c, n):  # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle]

def prepare_points(tensor, data_config, threshold = False):
    if threshold:
        tensor = np.where(
            tensor > data_config.threshold,
            data_config.lower,
            data_config.upper
        )
    tensor = tensor.reshape((
            tensor.shape[0],
            data_config.y_shape,
            data_config.x_shape,
            data_config.z_shape
        ))
    return tensor

class data_config:
    threshold = 0.2
    upper = 1
    lower = 0
    x_shape = 16
    y_shape = 16
    z_shape = 16

def get_dataset(data_path, args=None):
    channel = 1
    im_size = (16, 16, 16)
    num_classes = 10
    class_names = [str(c) for c in range(num_classes)]

    full_vectors = os.path.join(data_path, '3D-MNIST', 'full_dataset_vectors.h5')

    with h5py.File(full_vectors, "r") as hf:
        X_train = hf['X_train'][:]
        X_test = hf['X_test'][:]
        y_train = hf['y_train'][:]
        y_test = hf['y_test'][:]

    X_train = prepare_points(X_train, data_config)
    X_test = prepare_points(X_test, data_config)

    X_train, X_test = torch.tensor(X_train, dtype=torch.float, device=args.device), torch.tensor(X_test, dtype=torch.float, device=args.device)
    X_train, X_test = torch.unsqueeze(X_train, dim=1), torch.unsqueeze(X_test, dim=1)
    y_train, y_test = torch.tensor(y_train, dtype=torch.long, device=args.device), torch.tensor(y_test, dtype=torch.long, device=args.device)

    dst_train = TensorDataset(X_train, y_train)  # no augmentation
    dst_test = TensorDataset(X_test, y_test)
    testloader = torch.utils.data.DataLoader(dst_test, shuffle=False, batch_size=256, num_workers=0)

    return channel, im_size, num_classes, class_names, dst_train, dst_test, testloader


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def get_network(model, channel, num_classes, im_size=(32, 32)):
    #torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth = 64, 3

    if model == 'Conv3DNet':
        net = Conv3DNet(channel, num_classes, net_width, net_depth, im_size)
    else:
        net = None
        exit('unknown model: %s'%model)

    gpu_num = torch.cuda.device_count()
    if gpu_num>0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10

    elif ipc == 2:
        outer_loop, inner_loop = 1, 1
    elif ipc == 11:
        outer_loop, inner_loop = 10, 50
    elif ipc == 51:
        outer_loop, inner_loop = 50, 10
    
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop


def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug = False)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug = False)
    save_and_print(args.log_path, '%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    return net, acc_train, acc_test
