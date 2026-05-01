# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import copy
import torch
import random
import numpy as np
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
from .utils import WeightingHook
from torchvision import transforms
from sklearn.mixture import GaussianMixture
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.core.utils import get_data_loader
from semilearn.datasets.augmentation import RandAugment
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset

@IMB_ALGORITHMS.register('sage')
class SAGE(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(SAGE, self).__init__(args, net_builder, tb_log, logger)

        # warm_up epoch
        self.warm_up = args.warm_up

        # dataset update step
        if args.dataset == 'cifar10':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'cifar100':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'food101':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'svhn':
            self.update_step = 5
            self.memory_step = 5

        # augment r and dim
        self.sim_num = None
        self.feat_aug_r = None
        self.feat_dim = self.model.num_features

        # uniform class feature center
        self.optim_cfc = None
        self.cfcd = self.model.num_features

        # adaptive labeled (include the pseudo labeled) data and its dataloader
        self.current_x = None
        self.current_y = None
        self.current_idx = None
        self.current_noise_y = None
        self.current_one_hot_y = None
        self.current_one_hot_noise_y = None

        self.select_ulb_idx = None
        self.select_ulb_label = None
        self.select_ulb_pseudo_label = None
        self.select_ulb_pseudo_label_distribution = None

        self.adaptive_lb_dest = None
        self.adaptive_lb_dest_loader = None

        self.dataset = args.dataset
        self.data = self.dataset_dict['data']
        self.targets = self.dataset_dict['targets']
        self.noised_targets = self.dataset_dict['noised_targets']
        self.lb_idx =  self.dataset_dict['lb_idx']
        self.ulb_idx =  self.dataset_dict['ulb_idx']

        self.mean, self.std = {}, {}

        self.mean['cifar10'] = [0.485, 0.456, 0.406]
        self.mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
        self.mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
        self.mean['svhn'] = [0.4380, 0.4440, 0.4730]
        self.mean['food101'] = [0.485, 0.456, 0.406]

        self.std['cifar10'] = [0.229, 0.224, 0.225]
        self.std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
        self.std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
        self.std['svhn'] = [0.1751, 0.1771, 0.1744]
        self.std['food101'] = [0.229, 0.224, 0.225]

        if self.dataset == 'food101':
            self.transform_weak = transforms.Compose([
                                # transforms.Resize(args.img_size),
                                transforms.RandomCrop((args.img_size, args.img_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

            self.transform_strong = transforms.Compose([
                                # transforms.Resize(args.img_size),
                                transforms.RandomCrop((args.img_size, args.img_size)),
                                transforms.RandomHorizontalFlip(),
                                RandAugment(3, 5),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])
        else:
            self.transform_weak = transforms.Compose([
                                transforms.Resize(args.img_size),
                                transforms.RandomCrop(args.img_size, padding=int(args.img_size * (1 - args.crop_ratio)), padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

            self.transform_strong = transforms.Compose([
                                transforms.Resize(args.img_size),
                                transforms.RandomCrop(args.img_size, padding=int(args.img_size * (1 - args.crop_ratio)), padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(),
                                RandAugment(3, 5),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

        # compute lb dist
        lb_class_dist = [0 for _ in range(self.num_classes)]
        if args.noise_ratio > 0:
            for c in self.dataset_dict['train_lb'].noised_targets:
                lb_class_dist[c] += 1
        else:
            for c in self.dataset_dict['train_lb'].targets:
                lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)

        self.lb_dist = torch.from_numpy(lb_class_dist.astype(np.float32)).cuda(args.gpu)

        # compute select_ulb and ulb dist
        ulb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_ulb'].targets:
            ulb_class_dist[c] += 1
        ulb_class_dist = np.array(ulb_class_dist)

        self.ulb_dist = torch.from_numpy(ulb_class_dist.astype(np.float32)).cuda(args.gpu)

        self.select_ulb_dist = torch.zeros(self.num_classes).cuda(args.gpu)

        # compute lb_select_ulb and lb_ulb dist
        lb_ulb_class_dist = lb_class_dist + ulb_class_dist

        self.lb_ulb_dist = torch.from_numpy(lb_ulb_class_dist.astype(np.float32)).cuda(args.gpu)

        self.lb_select_ulb_dist = self.lb_dist + self.select_ulb_dist

        self.anchors = self.generate_simplex_prototypes(self.model.num_features + 1)

        WWT = self.anchors @ self.anchors.T
        I = torch.eye(self.model.num_features + 1, dtype=torch.float32).cuda(args.gpu)

        self.relations = torch.linalg.inv(0.1 * I + WWT)

    def generate_simplex_prototypes(self, num_prototypes):
        np.random.seed(self.args.seed)
        QR = np.random.random((self.model.num_features, num_prototypes))
        Q, R = np.linalg.qr(QR)
        Q = torch.from_numpy(Q.astype(np.float32)).cuda(self.args.gpu)
        R = torch.from_numpy(R.astype(np.float32)).cuda(self.args.gpu)
        O = (torch.eye(num_prototypes) - ((1 / num_prototypes) * torch.ones(num_prototypes, num_prototypes))).cuda(self.args.gpu)
        _, eig_vecs = torch.linalg.eigh(O)
        M = torch.sqrt(torch.tensor(num_prototypes / (num_prototypes - 1))) * torch.matmul(Q, eig_vecs[:, -self.model.num_features:].t())

        return M.T

    def simclr(self, feats_x_lb_w, feats_x_ulb_w, feats_x_ulb_s, pro_logits_x_lb_w, pro_logits_x_ulb_w, pro_logits_x_ulb_s, uratio, num_lb):
        lb_batch_size = num_lb
        ulb_batch_size = uratio * num_lb

        f_k = F.normalize(feats_x_lb_w, dim=1)
        f_i = F.normalize(feats_x_ulb_w, dim=1)
        f_j = F.normalize(feats_x_ulb_s, dim=1)
        z_k = F.normalize(pro_logits_x_lb_w, dim=1)
        z_i = F.normalize(pro_logits_x_ulb_w, dim=1)
        z_j = F.normalize(pro_logits_x_ulb_s, dim=1)

        relation_i = (z_i @ self.anchors.T) @ self.relations
        relation_j = (z_j @ self.anchors.T) @ self.relations

        if self.num_classes > 10:
            t_pred = 1.0
            t_graph = 0.5
            bias_val = 2.0
            step = 3
        else:
            t_pred = 0.5
            t_graph = 0.25
            bias_val = 2.0
            step = 5

        similarity_matrix_11 = torch.matmul(z_i, z_i.T) / t_pred
        similarity_matrix_21 = torch.matmul(z_i, z_j.T) / t_pred
        similarity_matrix_12 = torch.matmul(z_j, z_i.T) / t_pred
        similarity_matrix_22 = torch.matmul(z_j, z_j.T) / t_pred
        relation_11 = torch.matmul(relation_i, relation_i.T) / t_graph
        relation_21 = torch.matmul(relation_i, relation_j.T) / t_graph
        relation_12 = torch.matmul(relation_j, relation_i.T) / t_graph
        relation_22 = torch.matmul(relation_j, relation_j.T) / t_graph

        similarity_matrix = torch.cat([torch.cat([similarity_matrix_11, similarity_matrix_12], dim=1),
                                       torch.cat([similarity_matrix_21, similarity_matrix_22], dim=1)], dim=0)

        diag_idx = torch.arange(ulb_batch_size).cuda(self.args.gpu)
        relation_11[diag_idx, diag_idx] += bias_val
        relation_22[diag_idx, diag_idx] += bias_val
        relation_12[diag_idx, diag_idx] += bias_val
        relation_21[diag_idx, diag_idx] += bias_val
        relation = torch.cat([torch.cat([relation_11, relation_12], dim=1),
                              torch.cat([relation_21, relation_22], dim=1)], dim=0)

        mask = torch.eye(2 * ulb_batch_size, dtype=torch.bool).cuda(self.args.gpu)

        similarity_matrix = similarity_matrix[~mask].view(2 * ulb_batch_size, -1)

        P = F.softmax(relation, dim=1)

        targets = torch.matrix_power(P, step)
        targets = targets[~mask].view(2 * ulb_batch_size, -1)

        con_loss = F.binary_cross_entropy_with_logits(similarity_matrix, targets.detach())

        sim_loss = torch.mean(F.cosine_similarity(z_i, f_j.detach(), dim=1)) + torch.mean(F.cosine_similarity(z_j, f_i.detach(), dim=1))

        return con_loss, - sim_loss

    def set_hooks(self):
        super().set_hooks()
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(WeightingHook(num_classes=self.num_classes, n_sigma=self.args.n_sigma, momentum=self.args.ema_p, per_class=self.args.per_class), "MaskingHook")

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state()

            y_true = []
            y_pred = []
            y_conf = []

            conf_correct = [[] for _ in range(self.num_classes)]
            conf_incorrect = [[] for _ in range(self.num_classes)]

            conf_data = {'correct_true': [], 'correct_other': [], 'wrong_pred': [], 'wrong_true': [], 'wrong_other': []}

            draw = type(self.model)(first_stride=1, num_classes=self.num_classes)
            state_dict = {k: v.clone().detach() for k, v in self.model.state_dict().items()}
            draw.load_state_dict(state_dict)
            draw.cuda(self.args.gpu)
            draw.eval()
            with torch.no_grad():
                for data in self.loader_dict['eval_ulb']:
                    x = data['x_ulb_w']
                    y = data['y_ulb']

                    if isinstance(x, dict):
                        x = {k: v.cuda(self.gpu) for k, v in x.items()}
                    else:
                        x = x.cuda(self.gpu)
                    y = y.cuda(self.gpu)

                    logit = draw(x.detach())['logits']

                    prob = F.softmax(logit, dim=1)
                    pred = torch.max(logit, dim=1)[1].cpu().tolist()
                    conf = F.softmax(logit, dim=1).max(dim=1)[0].cpu().tolist()

                    for i in range(len(y.cpu().tolist())):
                        if pred[i] == y.cpu().tolist()[i]:
                            conf_correct[y.cpu().tolist()[i]].append(conf[i])

                            conf_data['correct_true'].append(prob[i].cpu().tolist()[y.cpu().tolist()[i]])
                            other_probs = np.delete(prob[i].cpu().tolist(), y.cpu().tolist()[i])
                            conf_data['correct_other'].extend([other_probs.mean()])
                        else:
                            conf_incorrect[pred[i]].append(conf[i])

                            conf_data['wrong_pred'].append(prob[i].cpu().tolist()[pred[i]])
                            conf_data['wrong_true'].append(prob[i].cpu().tolist()[y.cpu().tolist()[i]])
                            other_probs = np.delete(prob[i].cpu().tolist(), [y.cpu().tolist()[i], pred[i]])
                            conf_data['wrong_other'].extend([other_probs.mean()])

                    y_true.extend(y.cpu().tolist())
                    y_pred.extend(torch.max(logit, dim=1)[1].cpu().tolist())
                    y_conf.extend(F.softmax(logit, dim=1).max(dim=1)[0].cpu().tolist())
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_conf = np.array(y_conf)

            del draw
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state)

            if not os.path.exists(os.path.join(self.args.save_dir, self.args.save_name, str(self.epoch))):
                os.makedirs(os.path.join(self.args.save_dir, self.args.save_name, str(self.epoch)))

            if self.num_classes == 10:
                fig, axes = plt.subplots(self.num_classes, 2, figsize=(15, 5*self.num_classes), dpi=300)
                for cls in range(self.num_classes):
                    axes[cls, 0].hist(conf_correct[cls], bins=20, range=(0, 1), color='green', alpha=0.7)
                    axes[cls, 0].set_title(f'Class {cls} - Correct Predictions (Softmax)')
                    axes[cls, 0].set_xlabel('Confidence')
                    axes[cls, 0].set_ylabel('Count')
                    
                    axes[cls, 1].hist(conf_incorrect[cls], bins=20, range=(0, 1), color='red', alpha=0.7)
                    axes[cls, 1].set_title(f'Class {cls} - Incorrect Predictions (Softmax)')
                    axes[cls, 1].set_xlabel('Confidence')
                    axes[cls, 1].set_ylabel('Count')

                plt.tight_layout()
                plt.savefig(os.path.join(self.args.save_dir, self.args.save_name, str(self.epoch), 'confidence_dist_softmax.pdf'))
                plt.clf()
                plt.close()

            plt.figure(figsize=(15, 15))
            plt.suptitle(f'Confidence Distribution - Softmax Predictions', y=1.02)
            
            plt.subplot(3, 2, 1)
            plt.hist(conf_data['correct_true'], bins=50, range=(0, 1), alpha=0.7, 
                    label='True class', color='green')
            plt.hist(conf_data['correct_other'], bins=50, range=(0, 1), alpha=0.5, 
                    label='Other classes', color='blue')
            plt.title('Correct Predictions')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            plt.legend()
            
            plt.subplot(3, 2, 2)
            plt.hist(conf_data['wrong_pred'], bins=50, range=(0, 1), alpha=0.7, 
                    label='Predicted class', color='red')
            plt.hist(conf_data['wrong_true'], bins=50, range=(0, 1), alpha=0.7, 
                    label='True class', color='orange')
            plt.hist(conf_data['wrong_other'], bins=50, range=(0, 1), alpha=0.5, 
                    label='Other classes', color='purple')
            plt.title('Incorrect Predictions')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            plt.legend()
            
            plt.subplot(3, 2, 3)
            plt.boxplot([conf_data['correct_true'], conf_data['correct_other']], 
                       labels=['True class', 'Other classes'])
            plt.title('Correct Predictions (Boxplot)')
            plt.ylabel('Confidence')
            
            plt.subplot(3, 2, 4)
            plt.boxplot([conf_data['wrong_pred'], conf_data['wrong_true'], 
                       conf_data['wrong_other']], 
                      labels=['Predicted class', 'True class', 'Other classes'])
            plt.title('Incorrect Predictions (Boxplot)')
            plt.ylabel('Confidence')

            plt.subplot(3, 2, 5)
            sns.kdeplot(conf_data['correct_true'], color='green', label='Predicted class', bw_method=0.3, cumulative=True)
            sns.kdeplot(conf_data['correct_other'], color='blue', label='Other classes', bw_method=0.3, cumulative=True)
            plt.title('Correct Predictions')
            plt.xlabel('Confidence')
            plt.ylabel('Density')
            plt.legend()
            
            plt.subplot(3, 2, 6)
            sns.kdeplot(conf_data['wrong_pred'], color='red', label='Predicted class', bw_method=0.3, cumulative=True)
            sns.kdeplot(conf_data['wrong_true'], color='orange', label='True class', bw_method=0.3, cumulative=True)
            sns.kdeplot(conf_data['wrong_other'], color='purple', label='Other classes', bw_method=0.3, cumulative=True)
            plt.title('Incorrect Predictions')
            plt.xlabel('Confidence')
            plt.ylabel('Density')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.args.save_dir, self.args.save_name, str(self.epoch), f'confidence_dist_for_different_classes_softmax.pdf'))
            plt.clf()
            plt.close()

            self.print_fn(f"\nSoftmax Confidence Statistics:")
            self.print_fn(f"Correct - True class: {np.mean(conf_data['correct_true']):.4f} ± {np.std(conf_data['correct_true']):.4f}")
            self.print_fn(f"Correct - Other classes: {np.mean(conf_data['correct_other']):.4f} ± {np.std(conf_data['correct_other']):.4f}")
            self.print_fn(f"Incorrect - Predicted: {np.mean(conf_data['wrong_pred']):.4f} ± {np.std(conf_data['wrong_pred']):.4f}")
            self.print_fn(f"Incorrect - True class: {np.mean(conf_data['wrong_true']):.4f} ± {np.std(conf_data['wrong_true']):.4f}")
            self.print_fn(f"Incorrect - Other classes: {np.mean(conf_data['wrong_other']):.4f} ± {np.std(conf_data['wrong_other']):.4f}")

            if self.num_classes == 10:
                gmm_softmax = GaussianMixture(n_components=2, random_state=self.args.seed)
                data_softmax = y_conf.reshape(-1, 1)
                gmm_softmax.fit(data_softmax)

                means_softmax = gmm_softmax.means_.flatten()
                stds_softmax = np.sqrt(gmm_softmax.covariances_.flatten())
                weights_softmax = gmm_softmax.weights_.flatten()

                order_softmax = np.argsort(means_softmax)
                low_peak_softmax = (means_softmax[order_softmax][0], stds_softmax[order_softmax][0], weights_softmax[order_softmax][0])
                high_peak_softmax = (means_softmax[order_softmax][1], stds_softmax[order_softmax][1], weights_softmax[order_softmax][1])

                self.print_fn("\nSoftmax Bimodal Distribution Analysis:")
                self.print_fn(f"Low Confidence Peak: μ={low_peak_softmax[0]:.4f}, σ={low_peak_softmax[1]:.4f}, weight={low_peak_softmax[2]:.4f}")
                self.print_fn(f"High Confidence Peak: μ={high_peak_softmax[0]:.4f}, σ={high_peak_softmax[1]:.4f}, weight={high_peak_softmax[2]:.4f}")

                plt.figure(figsize=(10, 6))
                x = np.linspace(0, 1, 1000).reshape(-1, 1)
                logprob = gmm_softmax.score_samples(x)
                pdf = np.exp(logprob)

                plt.hist(y_conf, bins=50, density=True, alpha=0.7, color='red')
                plt.plot(x, pdf, '--k', label='Mixture')
                for i in range(2):
                    mean = gmm_softmax.means_[i, 0]
                    std = np.sqrt(gmm_softmax.covariances_[i, 0, 0])
                    weight = gmm_softmax.weights_[i]
                    component = weight * norm.pdf(x, mean, std)
                    plt.plot(x, component, label=f'Component {i+1}')

                plt.title('Softmax Confidence Bimodal Distribution')
                plt.xlabel('Confidence')
                plt.ylabel('Density')
                plt.legend()
                plt.savefig(os.path.join(self.args.save_dir, self.args.save_name, str(self.epoch), 'bimodal_softmax.pdf'))
                plt.close()

            class_indices = np.arange(self.num_classes)

            bar_width = 0.3
            index = class_indices - bar_width / 2

            record_mask_true = torch.zeros(self.num_classes).cpu()
            record_mask_false = torch.zeros(self.num_classes).cpu()

            record_mask_true.index_add_(0, torch.from_numpy(y_pred[y_pred==y_true]), torch.ones_like(torch.from_numpy(y_pred[y_pred==y_true]), dtype=record_mask_true.dtype))
            record_mask_false.index_add_(0, torch.from_numpy(y_pred[y_pred!=y_true]), torch.ones_like(torch.from_numpy(y_pred[y_pred!=y_true]), dtype=record_mask_false.dtype))

            self.print_fn('softmax_ulb_dist:\n' + np.array_str(np.array(self.ulb_dist.cpu())))
            self.print_fn('softmax_record_mask_true:\n' + np.array_str(np.array(record_mask_true)))
            self.print_fn('softmax_record_mask_false:\n' + np.array_str(np.array(record_mask_false)))
            self.print_fn('softmax_record_mask:\n' + np.array_str(np.array(record_mask_true + record_mask_false)))

            if self.num_classes == 10:
                fig = plt.figure(figsize=(8, 6), dpi=1000)

                ax = fig.add_subplot(111)

                bar0 = ax.bar(index, self.ulb_dist.tolist(), width=bar_width, color='#ffffff', edgecolor='black', label='GT')
                bar1 = ax.bar(index + bar_width, record_mask_true.tolist(), width=bar_width, color='#e37663', edgecolor='black', label='TP')
                for i in range(self.num_classes):
                    if i == 0:
                        bar2 = ax.bar(index[i] + bar_width, record_mask_false.tolist()[i], width=bar_width, bottom=record_mask_true.tolist()[i], color='#76a4bc', edgecolor='black', label='FP')
                    else:
                        bar2 = ax.bar(index[i] + bar_width, record_mask_false.tolist()[i], width=bar_width, bottom=record_mask_true.tolist()[i], color='#76a4bc', edgecolor='black')

                ax.set_ylim(0, max(max(np.array(self.ulb_dist.cpu())), max(np.array(record_mask_true + record_mask_false))) + 600)

                ax.set_xlabel('Class index', fontsize=18)
                ax.set_ylabel('Number of samples', fontsize=18)

                ax.set_xticks(class_indices)
                ax.set_xticklabels([f'{i}' for i in class_indices], fontsize=18, rotation=0)

                sample_indices = np.arange(0, max(max(np.array(self.ulb_dist.cpu())), max(np.array(record_mask_true + record_mask_false))) + 600, 1000)

                ax.set_yticks(sample_indices)
                ax.set_yticklabels([f'{int(i)}' for i in sample_indices], fontsize=18, rotation=0)

                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=2.5, handleheight=0.8, borderpad=0.2, columnspacing=0.8, fontsize=16, framealpha=0.2)

                plt.subplots_adjust(left=0.15, bottom=0.15)

                plt.savefig(os.path.join(self.args.save_dir, self.args.save_name, str(self.epoch), 'mask_true_false_softmax.pdf'))
                plt.clf()
                plt.close()

            # ~self.warm_up ce loss only and not select unlabeled data
            if self.epoch < self.warm_up:
                self.adaptive_lb_dest_loader = self.loader_dict['train_lb']
                self.lb_select_ulb_dist = self.lb_dist
                self.select_ulb_dist = torch.ones(self.num_classes).cuda(self.args.gpu)

            # self.warm_up select unlabeled data but still use labeled data only to compute loss
            elif self.epoch == self.warm_up:
                self.adaptive_lb_dest_loader = self.loader_dict['train_lb']
                self.lb_select_ulb_dist = self.lb_dist
                self.select_ulb_dist = torch.ones(self.num_classes).cuda(self.args.gpu)

            # self.warm_up+1~ use labeled (include the pseudo labeled) data and continue select unlabeled data
            # update the labeled (include the pseudo labeled) dataset and labeled (include the pseudo labeled) data distribution and selected unlabeled data distribution
            else:
                if (self.epoch - self.warm_up - self.memory_step) % self.update_step == 0:
                    self.print_fn(str(self.epoch) + ': Update the labeled data.')
                    # construct the current lb_select_ulb data and its dataloader
                    self.adaptive_lb_dest_loader = self.loader_dict['train_lb']
                    self.lb_select_ulb_dist = self.lb_dist
                    self.select_ulb_dist = torch.ones(self.num_classes).cuda(self.args.gpu)

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def train_step(self, idx_lb, x_lb_w, x_lb_s, y_lb, y_lb_noised, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]
        if self.args.noise_ratio > 0:
            lb = y_lb_noised
        else:
            lb = y_lb

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w, x_lb_s, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb_w, logits_x_lb_s = outputs['logits'][:2 * num_lb].chunk(2)
                aux_logits_x_lb_w, aux_logits_x_lb_s = outputs['aux_logits'][:2 * num_lb].chunk(2)
                pro_logits_x_lb_w, pro_logits_x_lb_s = outputs['pro_logits'][:2 * num_lb].chunk(2)
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][2 * num_lb:].chunk(2)
                aux_logits_x_ulb_w, aux_logits_x_ulb_s = outputs['aux_logits'][2 * num_lb:].chunk(2)
                pro_logits_x_ulb_w, pro_logits_x_ulb_s = outputs['pro_logits'][2 * num_lb:].chunk(2)
                feats_x_lb_w, feats_x_lb_s = outputs['feat'][:2 * num_lb].chunk(2)
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][2 * num_lb:].chunk(2)
            else:
                outs_x_lb_w = self.model(x_lb_w)
                logits_x_lb_w = outs_x_lb_w['logits']
                aux_logits_x_lb_w = outs_x_lb_w['aux_logits']
                pro_logits_x_lb_w = outs_x_lb_w['pro_logits']
                feats_x_lb_w = outs_x_lb_w['feat']
                outs_x_lb_s = self.model(x_lb_s)
                logits_x_lb_s = outs_x_lb_s['logits']
                aux_logits_x_lb_s = outs_x_lb_s['aux_logits']
                pro_logits_x_lb_s = outs_x_lb_s['pro_logits']
                feats_x_lb_s = outs_x_lb_s['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                aux_logits_x_ulb_s = outs_x_ulb_s['aux_logits']
                pro_logits_x_ulb_s = outs_x_ulb_s['pro_logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    aux_logits_x_ulb_w = outs_x_ulb_w['aux_logits']
                    pro_logits_x_ulb_w = outs_x_ulb_w['pro_logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb_w': feats_x_lb_w, 'x_lb_s': feats_x_lb_s, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            if self.epoch < self.warm_up:
                lb_smooth = torch.zeros(num_lb, self.num_classes).cuda(self.args.gpu)
                lb_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                lb_smooth.scatter_(1, lb.unsqueeze(1), 1.0 - self.args.smoothing)

                sup_loss = self.ce_loss(logits_x_lb_w + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)), lb_smooth, reduction='mean')

                aux_pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=self.compute_prob(logits_x_ulb_w.detach()), use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                aux_pseudo_label_w_smooth = torch.zeros(self.args.uratio * num_lb, self.num_classes).cuda(self.args.gpu)
                aux_pseudo_label_w_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                aux_pseudo_label_w_smooth.scatter_(1, aux_pseudo_label_w.unsqueeze(1), 1.0 - self.args.smoothing)
                mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=self.compute_prob(logits_x_ulb_w.detach()), softmax_x_ulb=False)
                aux_loss = self.consistency_loss(aux_logits_x_ulb_s, aux_pseudo_label_w_smooth, 'ce', mask=mask) + self.ce_loss(aux_logits_x_lb_w + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)), lb_smooth, reduction='mean')

                con_loss, sim_loss = self.simclr(feats_x_lb_w, feats_x_ulb_w, feats_x_ulb_s, pro_logits_x_lb_w, pro_logits_x_ulb_w, pro_logits_x_ulb_s, self.args.uratio, num_lb)

                total_loss = sup_loss + con_loss + sim_loss + aux_loss
            else:
                lb_smooth = torch.zeros(num_lb, self.num_classes).cuda(self.args.gpu)
                lb_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                lb_smooth.scatter_(1, lb.unsqueeze(1), 1.0 - self.args.smoothing)

                sup_loss = self.ce_loss(logits_x_lb_w + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)), lb_smooth, reduction='mean')

                aux_pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=self.compute_prob(logits_x_ulb_w.detach()), use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                aux_pseudo_label_w_smooth = torch.zeros(self.args.uratio * num_lb, self.num_classes).cuda(self.args.gpu)
                aux_pseudo_label_w_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                aux_pseudo_label_w_smooth.scatter_(1, aux_pseudo_label_w.unsqueeze(1), 1.0 - self.args.smoothing)
                mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=self.compute_prob(logits_x_ulb_w.detach()), softmax_x_ulb=False)
                aux_loss = self.consistency_loss(aux_logits_x_ulb_s, aux_pseudo_label_w, 'ce', mask=mask) + self.ce_loss(aux_logits_x_lb_w + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)), lb, reduction='mean')

                con_loss, sim_loss = self.simclr(feats_x_lb_w, feats_x_ulb_w, feats_x_ulb_s, pro_logits_x_lb_w, pro_logits_x_ulb_w, pro_logits_x_ulb_s, self.args.uratio, num_lb)

                total_loss = sup_loss + con_loss + sim_loss + aux_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         aux_loss=aux_loss.item(),
                                         con_loss=con_loss.item(),
                                         sim_loss=sim_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['prob_max_mu_t'] = self.hooks_dict['MaskingHook'].prob_max_mu_t.cpu()
        save_dict['prob_max_var_t'] = self.hooks_dict['MaskingHook'].prob_max_var_t.cpu()
        save_dict['prob_gap_mu_t'] = self.hooks_dict['MaskingHook'].prob_gap_mu_t.cpu()
        save_dict['prob_gap_var_t'] = self.hooks_dict['MaskingHook'].prob_gap_var_t.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].prob_max_mu_t = checkpoint['prob_max_mu_t'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_var_t = checkpoint['prob_max_var_t'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_gap_mu_t = checkpoint['prob_gap_mu_t'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_gap_var_t = checkpoint['prob_gap_var_t'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--warm_up', int, 30),
            SSL_Argument('--alpha', float, 1.0),
            SSL_Argument('--smoothing', float, 0.1),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--n_sigma', int, 2),
            SSL_Argument('--per_class', str2bool, False),
        ]