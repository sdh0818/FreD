import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import copy
import argparse
import numpy as np
import torch
from utils import get_loops, get_dataset, get_network, evaluate_synset, get_time, TensorDataset, epoch, set_seed, save_and_print, get_images

import shutil
from frequency_transforms import DCT

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
        self.num_per_class = int((self.ipc * self.im_size[0] * self.im_size[1] * self.im_size[2]) / self.msz_per_channel)

    def init(self, images_real, labels_real, indices_class):
        ### Initialize Frequency (F) ###
        images = torch.randn(size=(self.num_classes * self.num_per_class, self.channel, self.im_size[0], self.im_size[1], self.im_size[2]), dtype=torch.float, device=self.device)
        for c in range(self.num_classes):
            idx_shuffle = np.random.permutation(indices_class[c])[:self.num_per_class]
            images.data[c * self.num_per_class:(c + 1) * self.num_per_class] = images_real[idx_shuffle].detach().data
        self.freq_syn = self.dct.forward(images)
        self.freq_syn.requires_grad = True
        del images

        ### Initialize Mask (M) ###
        self.mask = torch.zeros(size=(self.num_classes * self.num_per_class, self.channel, self.im_size[0], self.im_size[1], self.im_size[2]), dtype=torch.float, device=self.device)
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
            for d, h, w in top_indices:
                self.mask[c * self.num_per_class:(c + 1) * self.num_per_class, :, d, h, w] = 1.0
            save_and_print(self.args.log_path, f"{get_time()} Class {c:3d} | {torch.sum(self.mask[c * self.num_per_class, 0] > 0.0):5d}")

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

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DM', help='DC/DSA/DM')
    parser.add_argument('--dataset', type=str, default='3D-MNIST', help='dataset')
    parser.add_argument('--model', type=str, default='Conv3DNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--eval_it', type=int, default=200)
    parser.add_argument('--epoch_eval_train', type=int, default=500, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    # parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='../data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')


    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sh_file', type=str)
    parser.add_argument('--FLAG', type=str, default="TEST")

    ### FreD ###
    parser.add_argument('--batch_syn', type=int)
    parser.add_argument('--msz_per_channel', type=int)
    parser.add_argument('--lr_freq', type=float)
    parser.add_argument('--mom_freq', type=float)
    args = parser.parse_args()
    set_seed(args.seed)

    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.save_path = args.save_path + f"/{args.FLAG}"
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    shutil.copy(f"./scripts/{args.sh_file}", f"{args.save_path}/{args.sh_file}")
    args.log_path = f"{args.save_path}/log.txt"

    eval_it_pool = np.arange(0, args.Iteration+1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, dst_train, dst_test, testloader = get_dataset(args.data_path, args=args)
    args.channel, args.im_size, args.num_classes = channel, im_size, num_classes
    model_eval_pool = ["Conv3DNet"]

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    for exp in range(args.num_exp):
        save_and_print(args.log_path, f'\n================== Exp {exp} ==================\n ')
        save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        ''' initialize the synthetic data '''
        synset = SynSet(args)
        synset.init(images_all, labels_all, indices_class)

        ''' training '''
        save_and_print(args.log_path, '%s training begins'%get_time())

        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    save_and_print(args.log_path, '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = synset.get(need_copy=True)
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)

                        del image_syn_eval, label_syn_eval
                    save_and_print(args.log_path, 'Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

            loss_avg = 0

            ''' update synthetic data '''
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                img_real = get_images(images_all, indices_class, c, args.batch_real)

                if args.batch_syn > 0:
                    indices = np.random.permutation(range(c * synset.num_per_class, (c + 1) * synset.num_per_class))[:args.batch_syn]
                else:
                    indices = range(c * synset.num_per_class, (c + 1) * synset.num_per_class)

                img_syn, lab_syn = synset.get(indices=indices)

                output_real = embed(img_real).detach()
                output_syn = embed(img_syn)

                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

            synset.optim_zero_grad()
            loss.backward()
            synset.optim_step()
            loss_avg += loss.item()

            loss_avg /= (num_classes)

            if it%10 == 0:
                save_and_print(args.log_path, '%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration: # only record the final results
                data_save.append([synset.get(need_copy=True)])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


    save_and_print(args.log_path, '\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        save_and_print(args.log_path, 'Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))


if __name__ == '__main__':
    main()


