# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:24:33 2022

@author: 24596
"""
import subprocess

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 22:28:24 2022

@author: volbem
"""

# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
import numpy as np
from gen_data import generate_dataset
from torch.utils.data import DataLoader
import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab as pl
from numpy import *
import tkinter as tk
import os
import webbrowser

torch.autograd.set_detect_anomaly(True)


class Zigmoid(nn.Module):
    def __init__(self, ):
        super(Zigmoid, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(1.0)

    def forward(self, input):
        return 1 / (0.5 + torch.exp(-self.weight * input))


class creatm_Loss(nn.Module):
    def __init__(self):
        super(creatm_Loss, self).__init__()

    def forward(self, x, y):
        loss = torch.mean(torch.pow(0.01, (x - y)))
        return loss


class WSNet_allo(torch.nn.Module):
    def __init__(self):  # 5个agent
        super(WSNet_allo, self).__init__()
        self.features = torch.nn.Sequential(  # 卷积
            torch.nn.Linear(8, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 1),

        )
        self.allonet = torch.nn.Sequential(torch.nn.Linear(5, 16),  # 支付
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Linear(16, 64),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Linear(64, 128),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Linear(128, 64),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Linear(64, 5),
                                           torch.nn.Softmax())  # 分配

    def forward(self, data):  # [batch_size,items,agents,type] [2,3,5,8]
        # data = data.to(torch.float32)
        '''
        x = self.features(data)
        x = x.reshape(data.shape[0],data.shape[1],data.shape[2])
        eitem_allo = self.allonet(x)
        
        '''
        eitem_allo = torch.torch.empty(0)
        for j in range(data.shape[1]):  # 第j个物品的拍卖 [2,1,5,8]
            agents = torch.torch.empty(0)
            for k in range(data.shape[2]):  # j物品的第k个agent [2,1,1,8]
                x = data[:, j, k, :]
                # x = x.view(-1,8)
                x = x.to(torch.float32)  # [2,]

                x = self.features(x).reshape(data.shape[0], 1, 1)  # [batch_size,anent,feature]

                agents = torch.cat((agents, x), 1)  # 第j个物品的k个agent的特征 [batch_size,agent,1]

            allo = self.allonet(agents.reshape(data.shape[0], -1))  # 第j个物品的k个agent的分配 #[batch_size,agent]
            allo = allo.reshape(data.shape[0], 1, data.shape[2])

            eitem_allo = torch.cat((eitem_allo, allo), 1)

        return eitem_allo


class WSNet_paym(torch.nn.Module):
    def __init__(self):  # 5个agent
        super(WSNet_paym, self).__init__()
        self.Zigmoid = Zigmoid()
        self.features = torch.nn.Sequential(  # 卷积
            torch.nn.Linear(8, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 1),
            #
        )
        self.paynet = torch.nn.Sequential(torch.nn.Linear(5, 16),  # 支付
                                          torch.nn.LeakyReLU(),
                                          torch.nn.Linear(16, 64),
                                          torch.nn.LeakyReLU(),
                                          torch.nn.Linear(64, 128),
                                          torch.nn.LeakyReLU(),
                                          torch.nn.Linear(128, 64),
                                          torch.nn.LeakyReLU(),
                                          torch.nn.Linear(64, 5)
                                          )

    def forward(self, data):
        # data = data.to(torch.float32)
        '''
        x = self.features(data)
        x = x.reshape(data.shape[0],data.shape[1],data.shape[2])
        
        paym = self.paynet(x)
        eitem_paym = self.Zigmoid(paym)
        
        '''
        eitem_paym = torch.torch.empty(0)
        for j in range(data.shape[1]):  # 第j个物品的拍卖 [2,j,5,8]
            agents = torch.torch.empty(0)
            for k in range(data.shape[2]):  # j物品的第k个agent [2,1,1,8]
                x = data[:, j, k, :]
                x = x.view(-1, 8)
                x = x.to(torch.float32)  # [2,]

                x = self.features(x).reshape(data.shape[0], 1, 1)  # [batch_size,anent,feature]

                agents = torch.cat((agents, x), 1)  # 第j个物品的k个agent的特征 [batch_size,agent,1]

            paym = self.paynet(agents.reshape(data.shape[0], -1))  # 第j个物品的k个agent的分配 #[batch_size,agent]
            paym = self.Zigmoid(paym)
            paym = paym.reshape(data.shape[0], 1, data.shape[2])

            eitem_paym = torch.cat((eitem_paym, paym), 1)

        return eitem_paym


class creat_mis(torch.nn.Module):
    def __init__(self):  # 5个agent
        super(creat_mis, self).__init__()
        self.create_m = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 8),
        )

    def forward(self, data):
        x = data
        misreport = self.create_m(x)
        '''misreport = torch.torch.empty(0)
        for j in range(data.shape[1]): #第j个物品的拍卖 [2,j,1,8]
            agents =torch.torch.empty(0)
            x = data[:,j,:]  #[2,3,8]
            #x = x.view(-1,8)   #[bs,8]
            x = x.to(torch.float32) 
            x = self.create_m(x).reshape(data.shape[0],data.shape[0],8)  #[bs,1,8] 
            misreport = torch.cat((misreport,x.reshape(data.shape[0],1,1,8)),1)    #[bs,1,1,8]->[bs,3,1,8]
        '''
        return misreport


def center_window(root, width=400, height=300):
    # 获取屏幕宽度和高度
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    # 计算窗口位置，使其位于屏幕中央
    alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    root.geometry(alignstr)


def start_training():
    model_allo = WSNet_allo()
    model_paym = WSNet_paym()
    train(model_allo, model_paym, epoch=200, learning_rate=0.001)


def creat_misreport(i, w, m, allbatch, allo_tru, paym_tru, model_allo, model_paym):
    creat = creat_mis()
    optimizer_mis = optim.SGD(creat.parameters(), lr=0.001, momentum=0.9)
    albatch = allbatch.clone()
    misbatch = albatch.clone()
    report = allbatch.clone()

    o_data = albatch[:, :, :, 0:8].float().clone()
    g_data = creat(o_data)
    g_data.requires_grad_(True)
    allo_m = model_allo(g_data)
    paym_m = model_paym(g_data)

    extra_r = torch.sum((w * (g_data[:, :, :, 1:8] - albatch[:, :, :, 1:8])), 3)  # [bs,3,1]
    misbatch[:, :, :, 0:8] = g_data.detach()
    misbatch[:, :, :, 8] = extra_r.detach()
    mis_ulti = allo_m * albatch[:, :, :, 0] - allo_m * paym_m * misbatch[:, :, :, 0] + allo_m * misbatch[:, :, :, 8]
    tru_ulti = allo_tru * albatch[:, :, :, 0] - allo_tru * paym_tru * albatch[:, :, :, 0]

    optimizer_mis.zero_grad()
    lossfun = creatm_Loss()
    loss_creat = lossfun(mis_ulti, tru_ulti)

    loss_creat.backward(retain_graph=True)
    optimizer_mis.step()

    report[:, :, m:m + 1, 0] = misbatch[:, :, m:m + 1, 0]
    report[:, :, m:m + 1, 1:8] = misbatch[:, :, m:m + 1, 1:8]
    report[:, :, m:m + 1, 8] = torch.sum((w * (misbatch[:, :, m:m + 1, 1:8] - allbatch[:, :, m:m + 1, 1:8])), 3)

    return report


def train(model_allo, model_paym, epoch, learning_rate):
    Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_allo.to(Device)
    model_paym.to(Device)

    num_agent = 5  # 参与人
    num_item = 3  # 拍卖物品
    num_example = 100  # 数据生成量
    test_num_example = 20  # 数据生成量
    batch_size = 5
    test_batch_size = 1
    reserve_price = torch.tensor([0.6]).to(torch.float32)  # 保留价
    reserve_type = torch.tensor([0.9]).to(torch.float32)  # 保留的非价格属性
    reserve_date = torch.tensor([8]).to(torch.float32)  # 保留的非价格的日期
    reserve = torch.tensor(
        [reserve_price, reserve_type, reserve_type, reserve_type, reserve_type, reserve_type, reserve_type,
         reserve_date])
    optimizer = optim.SGD([
        {'params': model_allo.parameters()},
        {'params': model_paym.parameters(), 'lr': 0.001}
    ], lr=learning_rate, momentum=0.9)

    # 记录用的空数组
    record_trainregret = []
    record_testregret = []
    record_train_auc_ulti = []
    record_test_auc_ulti = []
    record_trainloss = []
    record_testloss = []
    # 生成数据
    report = generate_dataset(num_agent, num_item, num_example)  # [0:8tru,8rev,9cost,10misallo,11truallo]
    testreport = generate_dataset(num_agent, num_item, test_num_example)
    traindata = DataLoader(report, batch_size, shuffle=True, num_workers=0)  # 5个agent， 3个item， 8个type
    testdata = DataLoader(testreport, test_batch_size, shuffle=True, num_workers=0)

    for i in range(epoch):
        count1 = 0
        for allbatch in traindata:  # 第j个batch，一个batch有两个拍卖
            count1 += 1

            # 高斯分布生成非价格属性权重
            wl = 0.07
            wu = 0.12
            w_mu, w_sigma = 0.08, 0.01
            wa = float(stats.truncnorm((wl - w_mu) / w_sigma, (wu - w_mu) / w_sigma, loc=w_mu, scale=w_sigma).rvs(
                1))  # auctioneer

            for m in range(allbatch.shape[2]):
                # 高斯分布生成非价格属性权重
                w = float(stats.truncnorm((wl - w_mu) / w_sigma, (wu - w_mu) / w_sigma, loc=w_mu, scale=w_sigma).rvs(
                    1))  # bidder

                c_batch = allbatch
                al1 = allbatch.tolist()
                inputdata = allbatch[:, :, :, 0:8].float().clone()

                # 输出在拍卖中真实出价的分配和支付结果
                allo_tru = model_allo(inputdata)
                paym_tru = model_paym(inputdata)

                # 投标人根据拍卖方的规则进行自适应地修改谎报的投标方案
                c_batch = creat_misreport(i, w, m, c_batch, allo_tru, paym_tru, model_allo, model_paym)  # [bs,3,1,8]

                mis_inputdata = c_batch[:, :, :, 0:8].float().clone()

                # 拍卖方根据投标人方案计算产生的额外成本   #如非价格属性太差导致拍卖方出现生产问题
                cost = torch.sub(reserve, c_batch[:, :, :, 0:8])
                c_batch[:, :, :, -1] = wa * torch.sum(cost, 3)
                cal_batch = c_batch.clone()

                sum_mis_ulti = torch.torch.empty(0)
                sum_tru_ulti = torch.torch.empty(0)
                all_auc_ulti = torch.torch.empty(0)

                # 输出在拍卖中虚假出价的分配和支付结果
                allo_mis = model_allo(mis_inputdata)
                paym_mis = model_paym(mis_inputdata)
                # 投标人真实和虚假出价的收益计算
                mis_ulti = allo_mis * (
                        allbatch[:, :, :, 0].view(batch_size, num_item, num_agent) + cal_batch[:, :, :, 8].view(
                    batch_size, num_item, num_agent)) - allo_mis * paym_mis * cal_batch[:, :, :, 0].view(batch_size,
                                                                                                         num_item,
                                                                                                         num_agent)
                tru_ulti = allo_tru * allbatch[:, :, :, 0].view(batch_size, num_item,
                                                                num_agent) - allo_tru * paym_tru * allbatch[:, :, :,
                                                                                                   0].view(batch_size,
                                                                                                           num_item,
                                                                                                           num_agent)
                # 投标人两种出价的收益落差
                regret = mis_ulti - tru_ulti
                # 计算平均值用于记录和网络更新
                mean_mis_ulti = torch.mean(mis_ulti)
                mean_tru_ulti = torch.mean(tru_ulti)
                mean_regret = mean_mis_ulti - mean_tru_ulti

                # 拍卖方的收益计算
                auc_ulti = torch.sum(allo_mis * paym_mis * (
                        cal_batch[:, :, :, 0].view(batch_size, num_item, num_agent) + cal_batch[:, :, :, -1].view(
                    batch_size, num_item, num_agent)), 2)  # 2,3,5
                # 计算平均值用于记录和网络更新
                mean_auc_ulti = torch.mean(auc_ulti)
                mean_auc_ultil = mean_auc_ulti.tolist()

                optimizer.zero_grad()

                # 计算损失值  打印出现在的拍卖规则在哪方面较差
                if mean_auc_ulti < reserve_price:
                    loss_func = creatm_Loss()
                    loss = loss_func(auc_ulti, reserve_price)
                    print("auc_ulti1")

                elif mean_tru_ulti < torch.tensor([0]).to(torch.float32):  #
                    loss_func = creatm_Loss()
                    loss = loss_func(mean_tru_ulti, torch.tensor([0]).to(torch.float32))
                    print("bidder_ulti")

                elif torch.abs(mean_regret) > 0.001:
                    loss_func = nn.L1Loss()
                    loss = loss_func(mis_ulti, tru_ulti)
                    print("regret")
                else:
                    loss_func = creatm_Loss()
                    loss = loss_func(auc_ulti, reserve_price)
                    print("auc_ulti2")

                loss.backward(retain_graph=True)
                optimizer.step()

                # 记录
                rec = loss_func(mis_ulti, tru_ulti).tolist()
                record_trainregret.append(mean_regret.tolist())  # 记录投标人的落差值
                record_trainloss.append(rec)  # 记录训练的损失
                record_train_auc_ulti.append(mean_auc_ultil)  # 记录训练的拍卖方收益

        test_loss, testulti, test_regret = test(i, model_allo, model_paym, testdata, test_batch_size)  # 开始测试
        record_testloss.extend(test_loss)  # 记录损失值

        record_test_auc_ulti.extend(testulti)  # 记录拍卖方收益
        record_testregret.extend(test_regret)  # 记录投标人的落差值

    plot_trainloss = []
    plot_testloss = []

    record_testulti = []
    record_trainulti = []

    record_test_regret = []
    record_train_regret = []

    # 减少记录数据规模
    for l in range(len(record_trainloss) // 4):
        plot_trainloss.append(mean(record_trainloss[l * 4:(l + 1) * 4]))
        plot_testloss.append(mean(record_testloss[l * 4:(l + 1) * 4]))

        record_trainulti.append(mean(record_train_auc_ulti[l * 4:(l + 1) * 4]))
        record_testulti.append(mean(record_test_auc_ulti[l * 4:(l + 1) * 4]))

        record_train_regret.append(mean(record_trainregret[l * 4:(l + 1) * 4]))
        record_test_regret.append(mean(record_testregret[l * 4:(l + 1) * 4]))

    fig = plt.figure(figsize=(30, 20))

    filename = "./model/nn-opt.pth"
    state = {'model_allo': model_allo.state_dict(),
             'model_paym': model_paym.state_dict()}
    torch.save(state, filename)

    p1 = pl.plot(plot_trainloss, 'mediumseagreen', label=u'train_loss')
    pl.legend()
    # 显示图例
    p2 = pl.plot(plot_testloss, 'coral', label=u'test_loss')
    pl.legend()
    pl.xlabel(u'iters')
    pl.ylabel(u'loss')

    np.savetxt("./loss/record_trainloss.txt", np.array(plot_trainloss))  # n表示用 2*sigmoid
    np.savetxt("./loss/record_testloss.txt", np.array(plot_testloss))

    np.savetxt("./record/record_trainulti.txt", np.array(record_trainulti))
    np.savetxt("./record/record_testulti.txt", np.array(record_testulti))

    np.savetxt("./record/record_trainregret.txt", np.array(record_train_regret))
    np.savetxt("./record/record_testregret.txt", np.array(record_test_regret))

    print("over")
    return 0


def test(i, model_allo, model_paym, testdata, batch_size):
    model_allo.eval()
    model_paym.eval()
    record_testloss = []
    record_auc_ulti = []
    record_testregret = []

    num_agent = 5
    num_item = 3

    reserve_price = torch.tensor([0.6]).to(torch.float32)
    reserve_type = torch.tensor([0.9]).to(torch.float32)
    reserve_date = torch.tensor([8]).to(torch.float32)
    reserve = torch.tensor(
        [reserve_price, reserve_type, reserve_type, reserve_type, reserve_type, reserve_type, reserve_type,
         reserve_date])

    for allbatch in testdata:

        for m in range(allbatch.shape[2]):
            wl = 0.07
            wu = 0.12
            w_mu, w_sigma = 0.08, 0.01
            w = float(
                stats.truncnorm((wl - w_mu) / w_sigma, (wu - w_mu) / w_sigma, loc=w_mu, scale=w_sigma).rvs(1))  # bidder
            wa = float(stats.truncnorm((wl - w_mu) / w_sigma, (wu - w_mu) / w_sigma, loc=w_mu, scale=w_sigma).rvs(
                1))  # auctioneer

            c_batch = allbatch
            al1 = allbatch.tolist()
            inputdata = allbatch[:, :, :, 0:8].float().detach()

            allo_tru = model_allo(inputdata)  # allo[3,5,8]
            paym_tru = model_paym(inputdata)

            c_batch = creat_misreport(i, w, m, allbatch, allo_tru, paym_tru, model_allo, model_paym)  # [bs,3,1,8]

            cost = torch.sub(reserve, c_batch[:, :, :, 0:8])

            mis_inputdata = c_batch[:, :, :, 0:8].float().detach()
            c_batch[:, :, :, -1] = wa * torch.sum(cost, 3)
            cal_batch = c_batch.detach()

            sum_mis_ulti = torch.torch.empty(0)
            sum_tru_ulti = torch.torch.empty(0)
            all_auc_ulti = torch.torch.empty(0)

            allo_mis = model_allo(mis_inputdata)
            paym_mis = model_paym(mis_inputdata)

            alllm = allo_mis.tolist()

            mis_ulti = allo_mis * (
                    allbatch[:, :, :, 0].view(batch_size, num_item, num_agent) + cal_batch[:, :, :, 8].view(
                batch_size, num_item, num_agent)) - allo_mis * paym_mis * cal_batch[:, :, :, 0].view(batch_size,
                                                                                                     num_item,
                                                                                                     num_agent)

            tru_ulti = allo_tru * allbatch[:, :, :, 0].view(batch_size, num_item,
                                                            num_agent) - allo_tru * paym_tru * allbatch[:, :, :,
                                                                                               0].view(batch_size,
                                                                                                       num_item,
                                                                                                       num_agent)

            regret = mis_ulti - tru_ulti
            regretl = regret.tolist()

            mean_mis_ulti = torch.mean(mis_ulti)
            mean_tru_ulti = torch.mean(tru_ulti)
            mean_regret = mean_mis_ulti - mean_tru_ulti

            auc_ulti = torch.sum(allo_mis * paym_mis * (
                    cal_batch[:, :, :, 0].view(batch_size, num_item, num_agent) + cal_batch[:, :, :, -1].view(
                batch_size, num_item, num_agent)), 2)  # 2,3,5

            auc_ulti1 = auc_ulti.tolist()
            auc_ulti = auc_ulti.to(torch.float32)

            mean_auc_ulti = torch.mean(auc_ulti)
            mean_auc_ultil = mean_auc_ulti.tolist()

            loss_func = nn.L1Loss()

            loss = loss_func(mis_ulti, tru_ulti)

            rec = loss.tolist()

            record_testregret.append(mean_regret.tolist())
            record_testloss.append(rec)
            record_auc_ulti.append(mean_auc_ultil)

    return record_testloss, record_auc_ulti, record_testregret


def open_loss_folder():
    try:
        subprocess.Popen(['open', 'loss'])
    except Exception as e:
        print(f"Error occurred: {e}")
def open_record_folder():
    try:
        subprocess.Popen(['open', 'record'])
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    root.title('多属性拍卖博弈算法')
    center_window(root)

    # 在窗口的中上位置放置一个标签
    label = tk.Label(root, text='珠澳协同智能多属性拍卖博弈算法')
    label.pack(pady=20)  # pady 参数可以增加上下的空白边距

    # 在下方添加一个按钮
    button = tk.Button(root, text='开始训练', command=start_training)
    button.pack()

    button2 = tk.Button(root, text='打开loss结果', command=open_loss_folder)
    button2.pack()
    button3 = tk.Button(root, text='打开record结果', command=open_record_folder)
    button3.pack()
    root.mainloop()
