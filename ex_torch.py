# ex for torch operation
import torch.nn as nn
import torch
from torch.nn import init
import torch.utils.data as data
import copy

# a simple learning process, every tensor has the grad attribute, update automatically.
# fc1_1 = nn.Linear(1,1)
# src = torch.Tensor([0.1])
# tar = torch.Tensor([10])
# optim = torch.optim.Adam(fc1_1.parameters(), lr=0.1)
# for i in range(10):
#     pred = fc1_1(src)
#     loss = torch.abs(pred - tar)
#     print('loop {} loss {}'.format(i, loss.item()))
#     optim.zero_grad()
#     loss.backward()
#     optim.step()

# dataset class test
# class TestDataset(data.Dataset):
#     def __len__(self):
#         """Return the total number of images in the dataset."""
#         return 10
#
#     def __getitem__(self, index):
#         return index
#
# ds1 = TestDataset()
# print(ds1.__getitem__(1))
#
# dl1 = torch.utils.data.DataLoader(
#     ds1       # inhirent from torch.dataset, so can be incooperated into loader
#            ) #
# for i,data in enumerate(dl1):
#     if i > 3:
#         break
#     print(data)
# for i,data in enumerate(dl1):   # still from beginning
#     if i > 5:
#         break
#     print(data)

############### Tensor test ###################
# ts1 = torch.randn(1,5,1,1)
# ts1 = torch.Tensor([
#     [1,5,8],
#     [2,3,9]
# ])
# ts3 = torch.ones(5,1,1)
# ts2 = torch.Tensor([3]).view(1,1,1,1)
# ts_scal1 = torch.randn(5, dtype=torch.float)
# ts4 = torch.ones(5,3,10,10)
# ts1 =  ts1.permute([1,0,2,3])
# print(ts1.shape)
# can we times with 1 dimension different
# print(ts1 * ts2)    # singleton can be timed together
# print(ts1)
# print(torch.cat((ts1,ts3)))   # first dimension
# print(ts2.size())
# ts3.squeeze_()  # in place operation nice
# print(ts3.shape)
# print(len(ts2.shape))   # subclass of turple, so can work
# acc_li = []
# for i in range(5):
#     rst = (ts1>5).sum().item()
# # print(rst, rst.item())
#     acc = rst/ts1.numel()
#     acc_li.append((acc))
# acc_ave= sum(acc_li)/len(acc_li)        #average of a list?   still valid, what is wrong will the abs_diff
# print('average', acc_ave)
# print(rst.item()/ts1.numel())
# ts_exp1 = ts_scal1[(None,)*3]     # only expand at 0 position
# print(ts_exp1.size())
# ts_exp = ts_scal1   # not affect the original
# for i in range(3):
#     ts_exp = ts_exp.unsqueeze(-1)
# print(ts_exp.expand_as(ts4))
# print(ts_scal1)

#### module test ##############
# mdl1 = torch.nn.Linear(3,1)
# print('before initialization weight', mdl1.weight)
# init.uniform(mdl1.weight)
# print('after initialization', mdl1.weight)
# with torch.no_grad():
#     mdl1.weight.data.fill_(0.1)
# print('set to 1', mdl1.weight)    # can be initialized.

# class TwoLayerNet(torch.nn.Module):
#     def __init__(self, D_in=2, H=3, D_out=1):
#         """
#         In the constructor we instantiate two nn.Linear modules and assign them as
#         member variables.
#         """
#         super(TwoLayerNet, self).__init__()
#         self.linear1 = torch.nn.Linear(D_in, H)
#         self.linear2 = torch.nn.Linear(H, D_out)
#
#     def forward(self, x):
#         """
#         In the forward function we accept a Tensor of input data and we must return
#         a Tensor of output data. We can use Modules defined in the constructor as
#         well as arbitrary operators on Tensors.
#         """
#         h_relu = self.linear1(x).clamp(min=0)
#         y_pred = self.linear2(h_relu)
#         return y_pred
# net1 = TwoLayerNet()
# print(type(net1))   # print out result is <class '__main__.TwoLayerNet'>, return is:  module.class_name

# keep empty tensor to save loss
# loss_sum =  torch.tensor(0)
# loss1 = torch.tensor(1.5)
# loss_sum_ref = loss_sum
# print('loss1 is', loss1)
# print('after sum loss_sum is', loss_sum)

# ts1 = torch.tensor([1,2])
# ts2 = ts1       # tensor is refereced with direct assign
# ts2 = ts1.clone()   # clone needed
# ts2[0] =5
# print("after assign ts1 is", ts1)
# print(ts1-1)      # can operate with normal scalar

# test the mean and sum
# ts1 = torch.tensor([[1,3,5],[1,4.,6]])
# ts2 = torch.tensor([[2,0,1],[2,4.,8]])
# meanSqt = (ts1.mean() - ts2.mean())**2
# sumSqt = (ts1.sum() -ts2.sum())**2
# print('mean sqt is', meanSqt)
# print('sum sqt is', sumSqt)
# print('scaling up mean is', (2*2)**2*meanSqt)
# ts1 = torch.randn([2,1,3,3])
# ts2 = torch.randn([2,1,3,3])
# print(ts1.mean([2,3]))  # reduce 2,3, dimension, give  2x1 tensor
# print(ts1.mean(3))  # collapse dim 1, get 2 vec.
#
ts2 = torch.tensor([1,1,9])
ts3 = torch.tensor([1,2,5])
dt1 = {'ts2':ts2, 'ts3':ts3}
# print('ts3 is', ts3)
# print('ts3 0 is', ts3[0])       # scalar tensor(1)
# print('ts3 0 is', ts3[0].numpy())       # scalar 1
dt1_cp = copy.deepcopy(dt1)
print(dt1_cp)



