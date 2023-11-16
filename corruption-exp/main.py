import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import time
import shutil
from utils import set_seed, get_source_dataset, get_target_dataset, get_network, ParamDiffAug, get_daparam, save_and_print, TensorDataset, epoch, get_time

np.set_printoptions(linewidth=250, suppress=True, precision=4)

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness',
                      'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

def main(args):

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    if args.method == "DC":
        args.dsa = False
    else:
        args.dsa = True

    if args.dsa:
        args.dc_aug_param = None
    else:
        args.dc_aug_param = get_daparam(args.source_dataset, "", "", args.ipc)

    save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')

    ''' organize the source dataset '''
    dst_train, images_train, labels_train = get_source_dataset(args)

    if args.target_dataset[-1] == "C": # CIFAR-10-C / ImageNet-Subset-C
        total_accs = np.zeros((args.num_eval, len(common_corruptions)))
    else: # CIFAR-10.1
        total_accs = np.zeros(args.num_eval)

    for idx_eval in range(args.num_eval):
        ''' Train the network from scratch'''
        net = get_network(args).to(args.device)
        images_train = images_train.to(args.device)
        labels_train = labels_train.to(args.device)
        lr = float(args.lr_net)
        Epoch = int(args.epoch_eval_train)
        lr_schedule = [Epoch // 2 + 1]
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        criterion = nn.CrossEntropyLoss().to(args.device)

        dst_train = TensorDataset(images_train, labels_train)
        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

        start = time.time()

        for ep in range(Epoch + 1):
            loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=True)
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        time_train = time.time() - start
        save_and_print(args.log_path, '%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f \n' % (get_time(), idx_eval, Epoch, int(time_train), loss_train, acc_train))

        ''' Test on the target dataset '''
        optimizer = None
        criterion = nn.CrossEntropyLoss().to(args.device)

        if args.target_dataset[-1] == "C": # CIFAR-10-C / ImageNet-Subset-C
            for idx_corruption, corruption in enumerate(common_corruptions):
                args.corruption = corruption
                dst_test, testloader = get_target_dataset(args)
                with torch.no_grad():
                    _, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)

                total_accs[idx_eval, idx_corruption] = acc_test
        else: # CIFAR-10.1
            dst_test, testloader = get_target_dataset(args)
            with torch.no_grad():
                _, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)

            total_accs[idx_eval] = acc_test

        save_and_print(args.log_path, '%s Evaluate_%02d: %s \n' % (get_time(), idx_eval, total_accs[idx_eval]))
        torch.save(net, f"{args.save_path}/net#{idx_eval}.pt")

    ''' Visualize and Save '''
    save_and_print(args.log_path , "=" * 100)
    save_and_print(args.log_path, f"Average across num_eval: {np.average(total_accs, axis=0)}")
    save_and_print(args.log_path, f"Average: {np.average(total_accs)}")
    torch.save(total_accs, f"{args.save_path}/total_accs.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default="FreD")
    parser.add_argument("--source_dataset", default='CIFAR10', type=str)
    parser.add_argument("--target_dataset", default='CIFAR10-C', type=str)
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument("--level", default=1, type=int)
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
    parser.add_argument('--synset_path', type=str, default='./trained_synset', help='trained synthetic dataset path')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sh_file', type=str)
    parser.add_argument('--FLAG', type=str, default="TEST")
    args = parser.parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.save_path = args.save_path + f"/{args.FLAG}"
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    shutil.copy(f"./scripts/{args.sh_file}", f"{args.save_path}/{args.sh_file}")
    args.log_path = f"{args.save_path}/log.txt"

    main(args)
