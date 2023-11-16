import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, set_seed, save_and_print, TensorDataset, get_images, epoch
import copy
import random
from reparam_module import ReparamModule

import shutil
import matplotlib.pyplot as plt
from frequency_transforms import DCT
from hyper_params import load_default

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class SynSet():
    def __init__(self, args):
        ### Basic ###
        self.args = args
        self.log_path = self.args.log_path
        self.channel = self.args.channel
        self.num_classes = self.args.num_classes
        self.im_size = self.args.im_size
        self.device = self.args.device
        self.ipc = self.args.ipc

        ### FreD ###
        self.dct = DCT(resolution=self.im_size[0], device=self.device)
        self.lr_freq = self.args.lr_freq
        self.mom_freq = self.args.mom_freq
        self.msz_per_channel = self.args.msz_per_channel
        self.num_per_class = int((self.ipc * self.im_size[0] * self.im_size[1]) / self.msz_per_channel)

    def init(self, images_real, labels_real, indices_class):
        ### Initialize Frequency (F) ###
        images = torch.randn(size=(self.num_classes * self.num_per_class, self.channel, self.im_size[0], self.im_size[1]), dtype=torch.float, device=self.device)
        for c in range(self.num_classes):
            idx_shuffle = np.random.permutation(indices_class[c])[:self.num_per_class]
            images.data[c * self.num_per_class:(c + 1) * self.num_per_class] = images_real[idx_shuffle].detach().data
        self.freq_syn = self.dct.forward(images)
        self.freq_syn.requires_grad = True
        del images

        ### Initialize Mask (M) ###
        self.mask = torch.zeros(size=(self.num_classes * self.num_per_class, self.channel, self.im_size[0], self.im_size[1]), dtype=torch.float, device=self.device)
        self.mask.requires_grad = False

        ### Initialize Label ###
        self.label_syn = torch.tensor([np.ones(self.num_per_class) * i for i in range(self.num_classes)], requires_grad=False, device=self.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
        self.label_syn = self.label_syn.long()

        ### Initialize Optimizer ###
        self.optimizers = torch.optim.SGD([self.freq_syn, ], lr=self.lr_freq, momentum=self.mom_freq)

        self.init_mask()
        self.optim_zero_grad()
        self.show_budget()

    def get(self, indices=None, need_copy=False):
        if not hasattr(indices, '__iter__'):
            indices = range(len(self.label_syn))

        if need_copy:
            freq_syn, label_syn = copy.deepcopy(self.freq_syn[indices].detach()), copy.deepcopy(self.label_syn[indices].detach())
            mask = copy.deepcopy(self.mask[indices].detach())
        else:
            freq_syn, label_syn = self.freq_syn[indices], self.label_syn[indices]
            mask = self.mask[indices]

        image_syn = self.dct.inverse(mask * freq_syn)
        return image_syn, label_syn

    def init_mask(self):
        save_and_print(self.args.log_path, "Initialize Mask")

        for c in range(self.num_classes):
            freq_c = copy.deepcopy(self.freq_syn[c * self.num_per_class:(c + 1) * self.num_per_class].detach())
            freq_c = torch.mean(freq_c, 1)
            freqs_flat = torch.flatten(freq_c, 1)
            freqs_flat = freqs_flat - torch.mean(freqs_flat, dim=0)

            try:
                cov = torch.cov(freqs_flat.T)
            except:
                save_and_print(self.args.log_path, f"Can not use torch.cov. Instead use np.cov")
                cov = np.cov(freqs_flat.T.cpu().numpy())
                cov = torch.tensor(cov, dtype=torch.float, device=self.device)
            total_variance = torch.sum(torch.diag(cov))
            vr_fl2f = torch.zeros((np.prod(self.im_size), 1), device=self.device)
            for idx in range(np.prod(self.im_size)):
                pc_low = torch.eye(np.prod(self.im_size), device=self.device)[idx].reshape(-1, 1)
                vector_variance = torch.matmul(torch.matmul(pc_low.T, cov), pc_low)
                explained_variance_ratio = vector_variance / total_variance
                vr_fl2f[idx] = explained_variance_ratio.item()

            v, i = torch.topk(vr_fl2f.flatten(), self.msz_per_channel)
            top_indices = np.array(np.unravel_index(i.cpu().numpy(), freq_c.shape)).T[:, 1:]
            for h, w in top_indices:
                self.mask[c * self.num_per_class:(c + 1) * self.num_per_class, :, h, w] = 1.0
            save_and_print(self.args.log_path, f"{get_time()} Class {c:3d} | {torch.sum(self.mask[c * self.num_per_class, 0] > 0.0):5d}")

        ### Visualize and Save ###
        indices_save = np.arange(10) * self.num_per_class
        grid = torchvision.utils.make_grid(self.mask[indices_save], nrow=10)
        plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
        plt.savefig(f"{self.args.save_path}/Mask.png", dpi=300)
        plt.close()

        mask_save = copy.deepcopy(self.mask.detach())
        torch.save(mask_save.cpu(), os.path.join(self.args.save_path, "mask.pt"))
        del mask_save

    def optim_zero_grad(self):
        self.optimizers.zero_grad()

    def optim_step(self):
        self.optimizers.step()

    def show_budget(self):
        save_and_print(self.log_path, '=' * 50)
        save_and_print(self.log_path, f"Freq: {self.freq_syn.shape} | Mask: {self.mask.shape} , {torch.sum(self.mask[0, 0] > 0.0):5d}")
        images, _ = self.get(need_copy=True)
        save_and_print(self.log_path, f"Decode condensed data: {images.shape}")
        del images
        save_and_print(self.log_path, '=' * 50)

def main(args):
    torch.autograd.set_detect_anomaly(True)

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    save_and_print(args.log_path, "CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    args.distributed = torch.cuda.device_count() > 1

    save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')
    save_and_print(args.log_path, f'Evaluation model pool: {model_eval_pool}')

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    save_and_print(args.log_path, "BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    ''' initialize the synthetic data '''
    synset = SynSet(args)
    synset.init(images_all, labels_all, indices_class)

    if args.batch_syn == 0:
        args.batch_syn = num_classes * synset.num_per_class

    ''' training '''
    syn_lr = torch.tensor(args.lr_teacher, device=args.device)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

    criterion = nn.CrossEntropyLoss().to(args.device)
    save_and_print(args.log_path, '%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset)
    if not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    save_and_print(args.log_path, "Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        save_and_print(args.log_path, "loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}
    del images_all, labels_all

    for it in range(0, args.Iteration+1):
        save_this_it = False

        ''' Evaluate synthetic data '''
        if it in eval_it_pool and it != 0:
            for model_eval in model_eval_pool:
                save_and_print(args.log_path, '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    save_and_print(args.log_path, f'DSA augmentation strategy: {args.dsa_strategy}')
                    save_and_print(args.log_path, f'DSA augmentation parameters: {args.dsa_param.__dict__}')
                else:
                    save_and_print(args.log_path, f'DC augmentation parameters: {args.dc_aug_param}')

                accs_test = []
                accs_train = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                    image_syn_eval, label_syn_eval = synset.get(need_copy=True)
                    save_and_print(args.log_path, f"Evaluate dataset size: {image_syn_eval.shape} {label_syn_eval.shape}")

                    args.lr_net = syn_lr.item()
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                save_and_print(args.log_path, 'Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                save_and_print(args.log_path, f"{args.save_path}")
                save_and_print(args.log_path, f"{it:5d} | Accuracy/{model_eval}: {acc_test_mean}")
                save_and_print(args.log_path, f"{it:5d} | Max_Accuracy/{model_eval}: {best_acc[model_eval]}")
                save_and_print(args.log_path, f"{it:5d} | Std/{model_eval}: {acc_test_std}")
                save_and_print(args.log_path, f"{it:5d} | Max_Std/{model_eval}: {best_std[model_eval]}")
                del image_syn_eval, label_syn_eval

            save_and_print(args.log_path, f"{it:5d} | Synthetic_LR: {syn_lr.detach().cpu()}")

        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save, label_save = synset.get(need_copy=True)
                save_dir = args.save_path

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                torch.save(label_save.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
                    torch.save(label_save.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    classes_save = np.random.permutation(num_classes)[:min(10, num_classes)]
                    indices_save = np.concatenate([c*synset.num_per_class+np.arange(min(10, synset.num_per_class)) for c in classes_save])
                    upsampled = upsampled[indices_save]
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=len(classes_save), normalize=True, scale_each=True)
                    plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
                    plt.savefig(f"{save_dir}/Synthetic_Images#{it}.png", dpi=300)
                    plt.close()

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        upsampled = upsampled[indices_save]
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=len(classes_save), normalize=True, scale_each=True)
                        plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
                        plt.savefig(f"{save_dir}/Clipped_Synthetic_Images#{it}.png", dpi=300)
                        plt.close()

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        upsampled = upsampled[indices_save]
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=len(classes_save), normalize=True, scale_each=True)
                        plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
                        plt.savefig(f"{save_dir}/Reconstructed_Images#{it}.png", dpi=300)
                        plt.close()

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            upsampled = upsampled[indices_save]
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=len(classes_save), normalize=True, scale_each=True)
                            plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)))
                            plt.savefig(f"{save_dir}/Clipped_Reconstructed_Images#{it}.png", dpi=300)
                            plt.close()

                    del image_save, label_save, upsampled

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        image_syn, label_syn = synset.get()
        syn_images = image_syn

        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()

            x = syn_images[these_indices]
            this_y = y_hat[these_indices]

            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)

        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        synset.optim_zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        synset.optim_step()
        optimizer_lr.step()

        syn_lr.data = syn_lr.data.clip(min=0.001) # To avoid invalid syn_lr (refer to HaBa)

        for _ in student_params:
            del _

        if it % 10 == 0:
            save_and_print(args.log_path, '%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=500, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=15000, help='how many distillation steps to perform')
    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'], help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='../data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='../buffers', help='buffer path')
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")
    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sh_file', type=str)
    parser.add_argument('--FLAG', type=str, default="TEST")
    parser.add_argument('--save_path', type=str, default="./results")

    parser.add_argument('--syn_steps', type=int)
    parser.add_argument('--expert_epochs', type=int)
    parser.add_argument('--max_start_epoch', type=int)
    parser.add_argument('--lr_lr', type=float)
    parser.add_argument('--lr_teacher', type=float)

    ### FreD ###
    parser.add_argument('--batch_syn', type=int)
    parser.add_argument('--msz_per_channel', type=int)
    parser.add_argument('--lr_freq', type=float)
    parser.add_argument('--mom_freq', type=float)
    args = parser.parse_args()
    set_seed(args.seed)
    args = load_default(args)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.save_path = args.save_path + f"/{args.FLAG}"
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    shutil.copy(f"./scripts/{args.sh_file}", f"{args.save_path}/{args.sh_file}")
    args.log_path = f"{args.save_path}/log.txt"

    main(args)