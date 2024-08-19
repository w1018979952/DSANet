from utils import *
import warnings
#from model import *
from model_source import Generator, feature_extractor, Dis, Class, class_sex, class_nation
warnings.filterwarnings('ignore')

from Dual_Enhancement import Dual_enhancement
import torch
import torch.nn as nn
import numpy as np
#from acsl import ACSL
from Amce import Amce
from Focal_Loss import focal_loss
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load(feature, D, G, C):
    states = torch.load('F2V6_0.7084974093264249.pkl')#
    feature.load_state_dict(states['feature'])
    G.load_state_dict(states['G'])
    D.load_state_dict(states['D'])
    C.load_state_dict(states['C'])
    return feature, D, G, C

def adjust_lr(optimizer, optimizer3, epoch):
    if 20 == epoch :
        for p1 in optimizer.param_groups:
            p1['lr'] = p1['lr'] * 0.1
        for p3 in optimizer3.param_groups:
            p3['lr'] = p3['lr'] * 0.1
    if 20 < epoch < 60 and epoch % 35 == 0:
        for p1 in optimizer.param_groups:
            p1['lr'] = p1['lr'] * 0.1
        for p3 in optimizer3.param_groups:
            p3['lr'] = p3['lr'] * 0.1


def train():
    d_conv_dim = 32
    c_conv_dim = 32
    g_conv_dim = 32
    feature = feature_extractor()
    feature_drop=Dual_enhancement()
    generator = Generator(g_conv_dim)
    Discri = Dis(d_conv_dim)
    Cls = Class(c_conv_dim)
    CLs_sex= class_sex()
    CLs_nat= class_nation()
    Rank_loss = lift_struct(1.2,1)
    Loss_Fc = nn.CrossEntropyLoss()
    Loss_Amce = Amce()
    Loss_focal = focal_loss()
    cuda = True if torch.cuda.is_available() else False
    #feature, Discri, generator, Cls = load(feature, Discri, generator, Cls)
    if cuda:
        feature = feature.to('cuda')
        feature_drop=feature_drop.to('cuda')
        generator = generator.to('cuda')
        Discri = Discri.to('cuda')
        Cls = Cls.to('cuda')
        Rank_loss = Rank_loss.to('cuda')
        Loss_Fc = Loss_Fc.to('cuda')
        Loss_Amce = Loss_Amce.to('cuda')
        Loss_focal = Loss_focal.to('cuda')
    batch_size = 50
    acc_best = 0
    MyDataSet = train_data_V2F()
    dataloader = DataLoader(MyDataSet, batch_size=batch_size, shuffle=False, num_workers=0)#num_workers=10
    optimizer1 = torch.optim.Adam(
        [
            {"params": feature.parameters(), "lr": 5e-2},
            {"params": generator.parameters(), "lr": 5e-3},
            {"params": Cls.parameters(), "lr": 5e-2},  
        ],
    )
    optimizer3 = torch.optim.Adam(Discri.parameters(), lr=5e-3, betas=(0.5, 0.999))

    for epoch in range(60):
        adjust_lr(optimizer1, optimizer3, epoch)
        feature.train()
        feature_drop.train()
        generator.train()
        Discri.train()
        Cls.train()
        count_train = 0.0  
        audio_count = 0.0  
        face_count = 0.0  
        total_train = 0.0  
        for i, data in enumerate(dataloader):
            f, a1, a2, label, face_m, audio_m = data
            f = f.to('cuda')
            a1 = a1.to('cuda')
            a2 = a2.to('cuda')
            label1=label
            label1 = label1.to('cuda')


            total_train += f.size(0)
            face_m = face_m.to('cuda')
            audio_m = audio_m.to('cuda')
            f, a1, a2 = feature(f, a1, a2)
            #f_drop, a1_drop, a2_drop = feature_drop(f, a1, a2)
            f_drop, a1_drop, a2_drop, f, a1, a2 = feature_drop(f, a1, a2)
            f_drop, a1_drop, a2_drop =  generator(f_drop, a1_drop, a2_drop)
            f, a1, a2 = generator(f, a1, a2)
            
            #######################################
            for p1 in Discri.parameters():
                p1.requires_grad = True
            for p2 in feature.parameters():
                p2.requires_grad = False
            for p3 in generator.parameters():
                p3.requires_grad = False
            for p4 in Cls.parameters():
                p4.requires_grad = False
           
            out1, out2, out3= Discri(f, a1, a2)
            dout1, dout2, dout3= Discri(f_drop, a1_drop, a2_drop)

            loss_d = 2 * Loss_Fc(out1, face_m) + Loss_Fc(out2, audio_m) + Loss_Fc(out3, audio_m)
            loss_dd = 2 * Loss_Fc(dout1, face_m) + Loss_Fc(dout2, audio_m) + Loss_Fc(dout3, audio_m)
            loss_d_total=loss_d + loss_dd

            optimizer3.zero_grad()
            loss_d_total.backward(retain_graph=True)
            for p in Discri.parameters():
                torch.nn.utils.clip_grad_norm(p.data, 0.02)
            optimizer3.step()

            audio_count += label_acc(out1, face_m)
            face_count += label_acc(out2, audio_m) + label_acc(out3, audio_m)
            #######################################
            for p1 in Discri.parameters():
                p1.requires_grad = False
            for p2 in feature.parameters():
                p2.requires_grad = True
            for p3 in generator.parameters():
                p3.requires_grad = True
            for p4 in Cls.parameters():
                p4.requires_grad = True

            out1, out2, out3 = Discri(f, a1, a2)
            dout1, dout2, dout3= Discri(f_drop, a1_drop, a2_drop)

            predict = Cls(f, a1, a2)  
            d_predict = Cls(f_drop, a1_drop, a2_drop)

            loss1_g = 2 * Loss_Fc(out1, audio_m) + Loss_Fc(out2, face_m) + Loss_Fc(out3, face_m)
            loss1_dg = 2 * Loss_Fc(dout1, audio_m) + Loss_Fc(dout2, face_m) + Loss_Fc(dout3, face_m)

            #loss_p = Loss_Fc(predict, label1)
            #loss_dp = Loss_Fc(d_predict, label1)
            loss_p = Loss_Amce(predict, label1)
            loss_dp = Loss_Amce(d_predict, label1)  
            #loss_p = Loss_focal(predict, label1)
            #loss_dp = Loss_focal(d_predict, label1)

            loss_m = compute_metric(label, Rank_loss, f, a1, a2)
            loss_dm = compute_metric(label, Rank_loss, f_drop, a1_drop, a2_drop)


            loss_total = loss1_g + 2 * loss_m + 3 * loss_p + loss1_dg + 2 * loss_dm + 3 * loss_dp
            
            count_train += label_acc(predict, label)
            if i % 10 == 0:
                print(epoch, i, 'G ', loss1_g.item(), ' M ', loss_m.item(), ' C ', loss_p.item(), 'D ', loss_d.item())
                print(epoch, i, 'Gd ', loss1_dg.item(), ' Md ', loss_dm.item(), ' Cd ', loss_dp.item(), 'Dd ', loss_dd.item())
                if count_train != 0:
                    print('counts =', count_train)

            optimizer1.zero_grad()
            loss_total.backward(retain_graph=True)
            for p3 in generator.parameters():
                torch.nn.utils.clip_grad_norm(p3.data, 0.02)
            optimizer1.step()
            
        audio_acc = audio_count / total_train
        face_acc = face_count / (total_train * 2)
        

        audio_acc = audio_count / total_train
        face_acc = face_count / (total_train * 2)
        acc = count_train / total_train
        print('epoch:', epoch, 'F2V training acc :', acc)
        print('Audio acc : ', audio_acc, 'Face acc : ', face_acc)
        acc_best = eval(feature, generator, Cls, Discri, epoch, acc_best) 

    print('training over')


if __name__ == '__main__':
    seed = 25
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train()
