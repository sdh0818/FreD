import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import time
import copy
import numpy as np
import torch
import torch.nn as nn
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, set_seed, save_and_print, get_images

import shutil
import torchvision
import matplotlib.pyplot as plt
from frequency_transforms import DCT

from glad_utils import build_dataset, get_eval_lrs, eval_loop

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

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.res, args=args)
    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    save_and_print(args.log_path, f'\n================== Exp {0} ==================\n ')
    save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')

    ''' organize the real dataset '''
    images_all, labels_all, indices_class = build_dataset(dst_train, class_map, num_classes)
    images_all, labels_all = images_all.to(args.device), labels_all.to(args.device)

    ''' initialize the synthetic data'''
    synset = SynSet(args)
    synset.init(images_all, labels_all, indices_class)

    ''' training '''
    criterion = nn.CrossEntropyLoss().to(args.device)
    save_and_print(args.log_path, '%s training begins'%get_time())

    best_acc = {"{}".format(m): 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    save_this_it = False
    for it in range(args.Iteration+1):

        if it in eval_it_pool:
            image_syn_eval, label_syn_eval = synset.get(need_copy=True)
            save_this_it = eval_loop(latents=image_syn_eval, f_latents=None, label_syn=label_syn_eval, G=None, best_acc=best_acc,
                                     best_std=best_std, testloader=testloader,
                                     model_eval_pool=model_eval_pool, channel=channel, num_classes=num_classes,
                                     im_size=im_size, it=it, args=args)

        ''' Train synthetic data '''
        net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width).to(args.device) # get a random model
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False

        embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

        loss_avg = 0

        ''' update synthetic data '''
        if 'BN' not in args.model: # for ConvNet
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                img_real = get_images(images_all, indices_class, c, args.batch_real)

                if args.batch_syn > 0:
                    indices = np.random.permutation(range(c * synset.num_per_class, (c + 1) * synset.num_per_class))[:args.batch_syn]
                else:
                    indices = range(c * synset.num_per_class, (c + 1) * synset.num_per_class)

                img_syn, lab_syn = synset.get(indices=indices)

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                output_real = embed(img_real).detach()
                output_syn = embed(img_syn)

                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

        else: # for ConvNetBN
            images_real_all = []
            images_syn_all = []
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real)

                if args.batch_syn > 0:
                    indices = np.random.permutation(range(c * synset.num_per_class, (c + 1) * synset.num_per_class))[:args.batch_syn]
                else:
                    indices = range(c * synset.num_per_class, (c + 1) * synset.num_per_class)

                img_syn, lab_syn = synset.get(indices=indices)

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                images_real_all.append(img_real)
                images_syn_all.append(img_syn)

            images_real_all = torch.cat(images_real_all, dim=0)
            images_syn_all = torch.cat(images_syn_all, dim=0)

            output_real = embed(images_real_all).detach()
            output_syn = embed(images_syn_all)

            loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)

        synset.optim_zero_grad()
        loss.backward()
        synset.optim_step()
        loss_avg += loss.item()

        loss_avg /= (num_classes)

        if it%10 == 0:
            save_and_print(args.log_path, '%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

        if it == args.Iteration: # only record the final results
            data_save.append([synset.get(need_copy=True)])
            torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%dipc.pt'%(args.dataset, args.model, args.ipc)))


if __name__ == '__main__':
    import shared_args

    parser = shared_args.add_shared_args()

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sh_file', type=str)
    parser.add_argument('--FLAG', type=str, default="TEST")

    ### FreD ###
    parser.add_argument('--batch_syn', type=int)
    parser.add_argument('--msz_per_channel', type=int)
    parser.add_argument('--lr_freq', type=float)
    parser.add_argument('--mom_freq', type=float)
    args = parser.parse_args()
    args.space = "p"
    args.zca = False

    set_seed(args.seed)

    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.save_path = args.save_path + f"/{args.FLAG}"
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    shutil.copy(f"./scripts/{args.sh_file}", f"{args.save_path}/{args.sh_file}")
    args.log_path = f"{args.save_path}/log.txt"

    main(args)


