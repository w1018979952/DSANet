import torch
import torch.nn as nn


class lift_struct(nn.Module):
    def __init__(self, alpha, multi):
        super(lift_struct, self).__init__()
        self.alpha = alpha
        self.multi = multi

    def forward(self, anchor, positive, neglist):
        batch = anchor.size(0)
        D_ij = torch.pairwise_distance(anchor, positive)
        D_in = torch.zeros(batch,self.multi)
        D_jn = torch.zeros(batch,self.multi)
        for i in range(self.multi):
            a = torch.pairwise_distance(anchor, neglist[i])
            D_in[:,i]= torch.exp(self.alpha - a)
            b = torch.pairwise_distance(positive, neglist[i])
            D_jn[:,i]= torch.exp(self.alpha - b)
        D_n = D_in.max(1)[0] + D_jn.max(1)[0]
        J = torch.log(D_n).to('cuda') + D_ij
        J = torch.clamp(J, min=0)
        loss = J.sum() / (2 * batch)
        return loss

class re_triplet(nn.Module):
    def __init__(self, margin):
        super(re_triplet, self).__init__()
        self.margin = margin
        self.loss = nn.TripletMarginLoss(self.margin)

    def forward(self, anchor, positive, n_list):
        loss = 0.0
        for i in range(len(n_list)):
            loss += self.loss(anchor, positive, n_list[i])
        loss = loss / len(n_list)
        return loss
        
"""
if __name__ == '__main__':
    L = lift_struct(1.0, 1)
    # L = RankList(1.2,0.4,1,2)
    # L = n_pair(2)
    anchor = torch.randn(64, 128)
    positive = torch.randn(64, 128)
    negative1 = torch.randn(64, 128)
    # negative2 = torch.randn(64,128)
    neglist = []
    # neglist.append(negative1)
    # neglist.append(negative2)
    loss = L(anchor, positive, negative1)
    print(loss)
    # print(loss)
    # count = distance_acc(anchor,positive,neglist)
    # print(count)
"""