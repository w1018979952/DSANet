import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Amce(nn.Module):

    def __init__(self, score_thr=0.1, loss_weight=1.0):
        
        super(Amce, self).__init__()

        self.score_thr = score_thr
        assert self.score_thr > 0 and self.score_thr < 1
        self.loss_weight = loss_weight


    def forward(self, cls_logits, labels):

        device = cls_logits.device
        self.n_i, self.n_c = cls_logits.size()
        target = cls_logits.new_zeros(self.n_i, self.n_c)
        weight = target

        unique_label = torch.unique(labels)
        with torch.no_grad():
            sigmoid_cls_logits = torch.sigmoid(cls_logits)
        sorted, indices = torch.sort(sigmoid_cls_logits)
        sigmoid_cls_sort = torch.gather(sigmoid_cls_logits, -1, indices)
        sort_max = sigmoid_cls_sort[:,-1]-self.score_thr
        distance=sigmoid_cls_logits-sort_max.repeat(self.n_c,1).T
        high_score_inds1=torch.nonzero(distance>0)
        weight_mask = torch.sparse_coo_tensor(high_score_inds1.t(), cls_logits.new_ones(high_score_inds1.shape[0]), size=(self.n_i, self.n_c), device=device).to_dense()


        for cls in unique_label:
            cls = cls.item()
            cls_inds = torch.nonzero(labels == cls).squeeze(1) 
            cur_labels = [cls]
            cur_labels = torch.tensor(cur_labels, device=cls_logits.device)
            tmp_label_vec = cls_logits.new_zeros(self.n_c)
            tmp_label_vec[cur_labels] = 1
            tmp_label_vec = tmp_label_vec.expand(cls_inds.numel(), self.n_c)
            target[cls_inds] = tmp_label_vec
            tmp_weight_mask_vec = weight_mask[cls_inds]
            tmp_weight_mask_vec[:, cur_labels] = 1
            weight_mask[cls_inds] = tmp_weight_mask_vec


        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, target.float(), reduction='none')
        return torch.sum(weight_mask * cls_loss) / self.n_i
