import os
import torch
from torch import nn
import numpy as np
import random



class Dual_enhancement(nn.Module):

    def __init__(self, r=4, c=4):
        super(Dual_enhancement, self).__init__()
        self.r = r
        self.c = c

    def forward(self, feature_v, feature_f1, feature_f2):
        N, C, h, w=feature_v.size()
        v1, add_v1 = Enhanceme_feature(self, feature_v.reshape(N, C, h*w))
        dv1 = v1.reshape(N, C, h, w)
        add1 = add_v1.reshape(N, C, h, w)

        v2, add_v2 = Enhanceme_feature(self, feature_f1.reshape(N, C, h*w))
        dv2 = v2.reshape(N, C, h, w)
        add2 = add_v2.reshape(N, C, h, w)

        v3, add_v3 = Enhanceme_feature(self, feature_f2.reshape(N, C, h*w))
        dv3= v3.reshape(N, C, h, w)
        add3 = add_v3.reshape(N, C, h, w)
        return dv1, dv2, dv3, add1, add2, add3
        

def helperb1(feature_map):
    row, col =np.where(feature_map.detach().cpu().numpy() == feature_map.detach().cpu().numpy().max())
    b1 = torch.zeros_like(feature_map)
    for i in range(len(row)):
        r, c = int(row[i]), int(col[i])
        b1[r, c] = 1
    return b1

def from_num_to_block(mat, r, c, num):
    assert len(mat.shape) == 2, ValueError("Feature map shape is wrong!")
    res = torch.zeros_like(mat)
    row, col = mat.shape
    block_r, block_c = int(row / r), int(col / c)
    index = np.arange(r * c) + 1
    index = index.reshape(r, c)
    [index_r, index_c] = np.argwhere(index == num)[0]

    if index_c + 1 == c:
        end_c = c + 1
    else:
        end_c = (index_c + 1) * block_c
    if index_r + 1 == r:
        end_r = r + 1
    else:
        end_r = (index_r + 1) * block_r

    res[index_r * block_r: end_r, index_c * block_c:end_c] = 1
    return res


            
def Enhanceme_feature(self,feature_maps):
    res_feature=torch.zeros_like(feature_maps)
    add_feature=torch.zeros_like(feature_maps)
    if len(feature_maps.shape) == 3:
        resb1 = []
        resb2 = []
        random_block = 0
        feature_maps_list = torch.split(feature_maps, 1)
        y1=torch.ones(feature_maps[0].shape).to('cuda')
        y0=torch.zeros(feature_maps[0].shape).to('cuda')
        
        for feature_map in feature_maps_list:
            feature_map = feature_map.squeeze()
            tmp = helperb1(feature_map)
            resb1.append(tmp)
            random_num=int(self.r * self.c * torch.rand(1))
            if random_num < 2:
                random_block = 1
            else:
                random_block = random_num-1

            tmp_1 = from_num_to_block(feature_map, self.r, self.c, random_block )
            tmp1 = torch.where(feature_map > torch.mean(feature_map), y1, feature_map)
            tmp0 = torch.where(tmp1 < 1, y0, tmp1)
            tmp_drop = torch.mul(tmp_1, tmp0)
            resb2.append(tmp_drop)
        
    elif len(feature_maps.shape) == 2:
        tmp = helperb1(feature_maps)
        random_num=int(self.r * self.c * torch.rand(1))
        if random_num < 2:
            random_block = 1
        else:
            random_block = random_num-1
        y1=torch.ones(feature_maps.shape)
        y0=torch.zeros(feature_maps.shape)
        tmp_1 = from_num_to_block(feature_maps, self.r, self.c, random_block )
        tmp1=torch.where(feature_maps > torch.max(feature_maps),y1, feature_maps)
        tmp0=torch.where(feature_maps < 1, tmp1, y0)
        tmp_drop = torch.mul(tmp_1, tmp0)
        resb1 = [tmp]
        resb2 = [tmp_drop]

    else:
        raise ValueError

    for x in range(len(resb1)):
        #index_block=torch.clamp(resb1[x] + resb2[x], 0, 1)
        index_block=torch.clamp(resb2[x], 0, 1)
        res_feature[x] = feature_maps[x] -  0.9*torch.mul(feature_maps[x],index_block)
        add_feature[x] = feature_maps[x] +  torch.mul(feature_maps[x],index_block)

    return res_feature, add_feature



"""
if __name__ == '__main__':
    feature_maps1 = torch.rand([5,32,10,10])
    feature_maps2 = torch.rand([5,32,10,10])
    feature_maps3 = torch.rand([5,32,10,10])
    #print("feature maps is: ", feature_maps,)
    db = DiversificationBlock()
    res1, res2, res3 = db(feature_maps1, feature_maps2, feature_maps3)
    print(res1[0], len(res1))

"""



