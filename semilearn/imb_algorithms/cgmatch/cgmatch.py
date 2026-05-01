# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import numpy as np
import pandas as pd
from collections import Counter
import torch.nn.functional as F
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.core.criterions import calculate_ece

from .utils import CGMatchThresholdingHook
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool, concat_all_gather
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score

@IMB_ALGORITHMS.register('cgmatch')
class CGMatch(ImbAlgorithmBase):

    """
        CGMatch algorithm.
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(CGMatch, self).__init__(args, net_builder, tb_log, logger)
        self.T = args.T
        self.use_hard_label = args.hard_label
        self.dist_align = args.dist_align
        self.dist_uniform = args.dist_uniform
        self.num_data = self.args.ulb_dest_len
        self.queue_size = args.queue_size
        self.warm_up_iter = args.warm_up_iter
        self.guid_gold = {}
        # max_probs_queue: max prediction probs for each unlabeled data as training proceeds
        # pseudo_labels_queue: pseudo labels for each unlabeled data as training proceeds
        self.queue_max_probs = torch.zeros((self.num_data, self.queue_size)).cuda(self.gpu)
        self.queue_pseudo_labels = torch.zeros((self.num_data, self.queue_size), dtype=torch.long).cuda(self.gpu)
        self.queue_ptr = torch.zeros(self.num_data, dtype=torch.long).cuda(self.gpu)
        self.confidence_ = {}
        self.variability_ = {}
        self.correctness_ = {}
        self.count_gap = {}
        self.count_gap_acc = {}
        self.model_dir = args.save_dir + '/' + args.save_name + '/'
        self.lambda_h = args.lambda_h
        # paras for dynamic threshold of max_probs
        self.ema_p = args.ema_p
        self.n_sigma = args.n_sigma
        self.per_class = args.per_class

        self.ulb_idx =  self.dataset_dict['ulb_idx']

    def set_hooks(self):
        super().set_hooks()
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(CGMatchThresholdingHook(num_classes=self.num_classes, n_sigma=self.args.n_sigma, momentum=self.args.ema_p, per_class=self.args.per_class), "MaskingHook")
        self.register_hook(DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='uniform' if self.args.dist_uniform else 'model'), "DistAlignHook")

    #  update memory bank
    @torch.no_grad()
    def update_bank(self, ulb_idxs, pseudo_labels, y_max_probs):
        if self.distributed and self.world_size > 1:
            ulb_idxs = concat_all_gather(ulb_idxs)
            pseudo_labels = concat_all_gather(pseudo_labels)
            y_max_probs = concat_all_gather(y_max_probs)

        self.queue_max_probs[ulb_idxs, self.queue_ptr[ulb_idxs]] = y_max_probs
        self.queue_pseudo_labels[ulb_idxs, self.queue_ptr[ulb_idxs]] = pseudo_labels
        self.queue_ptr[ulb_idxs] = (self.queue_ptr[ulb_idxs] + 1) % self.queue_size

    @torch.no_grad()
    def compute_count_gap(self, ulb_idxs=None):
        if ulb_idxs is not None:
            pseudo_labels_queue = self.queue_pseudo_labels[ulb_idxs]
        else:
            pseudo_labels_queue = self.queue_pseudo_labels

        max_values = []
        count_gaps = []
        for row in pseudo_labels_queue.cpu().tolist():
            counter = Counter(row)

            max_value, max_count = counter.most_common(1)[0]

            sorted_counts = sorted(counter.values(), reverse=True)
            second_max_count = sorted_counts[1] if len(sorted_counts) > 1 else 0

            difference = max_count - second_max_count

            max_values.append(max_value)
            count_gaps.append(difference)

        return count_gaps, max_values

    @torch.no_grad()
    def compute_train_dy_metrics(self, ulb_idxs=None):
        if ulb_idxs is not None:
            if self.distributed and self.world_size > 1:
                ulb_idxs = concat_all_gather(ulb_idxs)
            pseudo_labels_queue = self.queue_pseudo_labels[ulb_idxs]
            max_probs_queue = self.queue_max_probs[ulb_idxs]
            ulb_idxs_list = ulb_idxs.cpu().tolist()
        else:
            pseudo_labels_queue = self.queue_pseudo_labels
            max_probs_queue = self.queue_max_probs
            ulb_idxs_list = range(self.num_data)

        # Compute variability for each unlabeled data
        var = torch.std(max_probs_queue, dim=1)
        # Compute confidence for each unlabeled data
        conf = torch.mean(max_probs_queue, dim=1)
        count_gaps, max_values = self.compute_count_gap(ulb_idxs)

        for i, guid in enumerate(ulb_idxs_list):
            self.confidence_[guid] = conf[i].item()
            self.variability_[guid] = var[i].item()
            self.count_gap[guid] = count_gaps[i]
            self.correctness_[guid] = torch.sum(pseudo_labels_queue[i] == self.guid_gold[guid]['gold']).item() / self.queue_size
            if guid in self.guid_gold:
                self.count_gap_acc[guid] = 1 if self.guid_gold[guid]['gold'] == max_values[i] else 0
            else:
                self.count_gap_acc[guid] = 0

        return count_gaps

    @torch.no_grad()
    def save_train_dy_metrics(self):
        _ = self.compute_train_dy_metrics()

        column_names = ['guid', 'index', 'confidence', 'variability', 'correctness', 'count_gap', 'count_gap_acc',]
        train_dy_metrics = pd.DataFrame([[guid, i, self.confidence_[guid], self.variability_[guid], self.correctness_[guid], 
                                          self.count_gap[guid], self.count_gap_acc[guid]] for i, guid in enumerate(self.correctness_)], columns=column_names)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        train_dy_filename = os.path.join(self.model_dir, f"td_metrics.jsonl")
        # save queue_max_probs and queue_pseudo_labels
        queue_max_probs_filename = os.path.join(self.model_dir, f"queue_max_probs.pt")
        queue_pseudo_labels_filename = os.path.join(self.model_dir, f"queue_pseudo_labels.pt")
        train_dy_metrics.to_json(train_dy_filename, orient='records', lines=True)
        torch.save(self.queue_max_probs, queue_max_probs_filename)
        torch.save(self.queue_pseudo_labels, queue_pseudo_labels_filename)

    def warm_up(self, idx_lb, x_lb_w, x_lb_s, y_lb, y_lb_noised):
        with self.amp_cm():
            outputs = self.model(x_lb_w)
            logits_x_lb = outputs['logits']
            feats_x_lb = outputs['feat']

            feat_dict = {'x_lb':feats_x_lb}

            total_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(total_loss=total_loss.item(), prefix='warm_up_train')

        return out_dict, log_dict

    @torch.no_grad()
    def nonlinear_weight_tracker(self):
        num_steps = float(self.it - self.warm_up_iter)
        num_steps = num_steps / float(self.num_train_iter - self.warm_up_iter)
        current_weight = self.lambda_h * (num_steps ** 2)

        return current_weight

    def train_step(self, idx_lb, x_lb_w, x_lb_s, y_lb, y_lb_noised, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        all_ulb_idx = np.array(self.ulb_idx)
        batch_ulb_idx = np.array(idx_ulb.cpu())
        positions = torch.from_numpy(np.where(all_ulb_idx[:, None] == batch_ulb_idx)[0])
        idx_ulb = positions.to(idx_ulb)

        num_lb = y_lb.shape[0]
        if self.args.noise_ratio > 0:
            lb = y_lb_noised
        else:
            lb = y_lb

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb_w) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())
            max_probs, _ = torch.max(probs_x_ulb_w, dim=-1)
            
            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=probs_x_ulb_w, use_hard_label=self.use_hard_label, T=self.T, softmax=False)
            
            self.update_bank(idx_ulb, pseudo_label, max_probs)

            # compute and update metrics
            count_gaps = self.compute_train_dy_metrics(idx_ulb)

            # compute mask
            easy_mask, ambiguous_mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, idx_ulb=idx_ulb, softmax_x_ulb=False)

            easy_unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=easy_mask)

            ambiguous_unsup_loss = self.gce_loss(logits_x_ulb_w, pseudo_label, mask=ambiguous_mask) + self.gce_loss(logits_x_ulb_s, pseudo_label, mask=ambiguous_mask)

            current_weight = self.nonlinear_weight_tracker()

            total_loss = sup_loss + self.lambda_u * easy_unsup_loss + current_weight * ambiguous_unsup_loss

            # calculate calibration metrics
            ece = calculate_ece(logits_x_ulb_w.detach().cpu(), y_ulb.detach().cpu())

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         ave_max_probs=max_probs.mean().item(),
                                         easy_unsup_loss=easy_unsup_loss.item(),
                                         ambiguous_unsup_loss=ambiguous_unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         easy_util_ratio=easy_mask.float().mean().item(),
                                         ambiguous_util_ratio=ambiguous_mask.float().mean().item(),
                                         ave_count_gaps=sum(count_gaps) / len(count_gaps),
                                         ece=ece)
        return out_dict, log_dict

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.loader_dict["train_lb"], self.loader_dict["train_ulb"]):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")

                if self.it < self.warm_up_iter:
                    self.out_dict, self.log_dict = self.warm_up(**self.process_batch(**data_lb))
                else:
                    self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))

                self.call_hook("after_train_step")
                self.it += 1

                if self.it == self.warm_up_iter:
                    self.save_train_dy_metrics()

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        """
        evaluation function
        """
        self.model.eval()
        self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_probs = []
        y_logits = []
        with torch.no_grad():
            for data in eval_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                logits = self.model(x)[out_key]

                loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                y_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
                total_loss += loss.item() * num_batch
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_probs, k=5)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        ece = calculate_ece(torch.tensor(y_logits), torch.tensor(y_true))

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()

        eval_dict = {eval_dest + '/loss': total_loss / total_num, eval_dest + '/top-1-acc': top1,
                     eval_dest + '/top-5-acc': top5, eval_dest + '/balanced_acc': balanced_top1,
                     eval_dest + '/precision': precision, eval_dest + '/recall': recall, eval_dest + '/F1': F1,
                     eval_dest + "/ece": ece}
        if return_logits:
            eval_dict[eval_dest + '/logits'] = y_logits
        return eval_dict

    def warm_up_evaluate(self, eval_dest='eval_ulb', out_key='logits'):
        """
        evaluation function
        """
        self.model.eval()
        self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        ulb_true_labels = []
        pseudo_labels = []
        ulb_idxs = []
        y_max_probs = []
        with torch.no_grad():
            for data in eval_loader:
                idx = data['idx_ulb']
                x = data['x_ulb_w']
                y = data['y_ulb']

                all_ulb_idx = np.array(self.ulb_idx)
                batch_ulb_idx = np.array(idx)
                positions = torch.from_numpy(np.where(all_ulb_idx[:, None] == batch_ulb_idx)[0])
                idx = positions.to(idx)

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                logits = self.model(x)[out_key]
                probs = torch.softmax(logits, dim=-1)

                ulb_true_labels.extend(y.cpu().tolist())
                pseudo_labels.extend(torch.max(probs, dim=-1)[1].cpu().tolist())
                ulb_idxs.extend(idx.cpu().tolist())
                y_max_probs.extend(torch.max(probs, dim=-1)[0].cpu().tolist())

        y_true = np.array(ulb_true_labels)
        ulb_idxs = torch.tensor(ulb_idxs).cuda(self.gpu)
        pseudo_labels = torch.tensor(pseudo_labels).cuda(self.gpu)
        y_max_probs = torch.tensor(y_max_probs).cuda(self.gpu)

        # fill the initial memory bank
        for i, idx in enumerate(ulb_idxs):
            if idx.item() not in self.guid_gold:
                self.guid_gold[idx.item()] = {"gold": y_true[i].item()}
        self.update_bank(ulb_idxs, pseudo_labels, y_max_probs)

        self.ema.restore()
        self.model.train()

        warm_up_eval_dict = {eval_dest + "/ave_max_probs": y_max_probs.mean().item()}
        return warm_up_eval_dict

    # TODO: change these
    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        if self.hooks_dict['DistAlignHook'].p_model is not None:
            save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        else:
            save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model 
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        save_dict['queue_ptr'] = self.queue_ptr.cpu()
        save_dict['queue_max_probs'] = self.queue_max_probs.cpu()
        save_dict['queue_pseudo_labels'] = self.queue_pseudo_labels.cpu()
        save_dict['guid_gold'] = self.guid_gold
        save_dict['confidence_'] = self.confidence_ 
        save_dict['variability_'] = self.variability_ 
        save_dict['correctness_'] = self.correctness_ 
        save_dict['count_gap'] = self.count_gap
        save_dict['count_gap_acc'] = self.count_gap_acc
        save_dict['prob_max_mu_t'] = self.hooks_dict['MaskingHook'].prob_max_mu_t.cpu()
        save_dict['prob_max_var_t'] = self.hooks_dict['MaskingHook'].prob_max_var_t.cpu()
        if self.hooks_dict['MaskingHook'].count_gap_mu_t is not None:
            save_dict['count_gap_mu_t'] = self.hooks_dict['MaskingHook'].count_gap_mu_t.cpu()
        else:
            save_dict['count_gap_mu_t'] = self.hooks_dict['MaskingHook'].count_gap_mu_t
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        if checkpoint['p_model'] is not None:
            self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        else:
            self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model']
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.queue_max_probs = checkpoint['queue_max_probs'].cuda(self.gpu)
        self.queue_pseudo_labels = checkpoint['queue_pseudo_labels'].cuda(self.gpu)
        self.queue_ptr = checkpoint['queue_ptr'].cuda(self.gpu)
        self.guid_gold = checkpoint['guid_gold']
        self.confidence_ = checkpoint['confidence_']
        self.variability_ = checkpoint['variability_']
        self.correctness_ = checkpoint['correctness_']
        self.count_gap = checkpoint['count_gap']
        self.count_gap_acc = checkpoint['count_gap_acc']
        self.hooks_dict['MaskingHook'].prob_max_mu_t = checkpoint['prob_max_mu_t'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_var_t = checkpoint['prob_max_var_t'].cuda(self.args.gpu)
        if checkpoint['count_gap_mu_t'] is not None:
            self.hooks_dict['MaskingHook'].count_gap_mu_t = checkpoint['count_gap_mu_t'].cuda(self.args.gpu)
        else:
            self.hooks_dict['MaskingHook'].count_gap_mu_t = checkpoint['count_gap_mu_t']
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--n_sigma', int, 2),
            SSL_Argument('--per_class', str2bool, False),
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--dist_uniform', str2bool, True),
            SSL_Argument('--queue_size', int, 1000),
            SSL_Argument('--warm_up_iter', int, 2048),
            SSL_Argument('--lambda_h', float, 1.0),
        ]
