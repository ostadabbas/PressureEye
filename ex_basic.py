# test basics
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

def func1():
    func2()
    # print('func1')

def func2():
    print('haha')


##### basic
# print(5%1) # 0
# print(5%-2) # -1  5 - (-2)* (-3) = -1
# print(5%-1) # 0
# print(5%-2) #
# li1 = [1, 2, 3]
# li1.insert(0, 'haha')
# print(li1)



#### other
# func1()     # early func can recognize the later ones
# li_train = range(5)
# print(li_train)
# for i in li_train:
#     print(i)
# li1  = [2,5,7]
# li2 = ['hah', 1.3]
# print(li1+li2)
# a,b,c = li1
# print(a,b,c)
# strRst = '_'.join(str(e))
# print('_'.join(str(e) for e in li1+li2))
# li3 = list(map(lambda x: x+1, li1))
# print(li3)
# arr1 = np.array([
#     [1,2,-5],
#     [6,-9,-1]
# ]).astype(float)
# # np.float32
# arr2 = np.zeros(5)      # default np64
# print(arr2.dtype)
# if np.float32 == arr1.dtype:
#     print('is float')
# else:
#     print('not equal')

# ts1 = torch.tensor([2,5,8])
# print(ts1.dtype)

# if np.float_==np.float64:
#     print('euqal float to 32')
# print(np.uint8(123.2))  # 123
# a = np.uint8(123.2)
# print(type(a))
# print('a is uint8 {}'.format(type(a) == np.uint8))
# print('-123.2 cast',  np.uint8(-123.2))  #  -> 133, negative sign make a difference .
# a = np.float(123.2)       #
# print(type(a))
# print('same type? {}'.format(type(a) == float))
#
# a = np.float64(123.2)  #
# print(type(a))
# print('same type? {}'.format(type(a) == float))

############ numpy test
# arr_empty = np.array([])
# arr2 = np.random.rand(2,3,2)
# arr3 = np.ones([4,5,2])
# arr2 = np.ones([3,1])
# arr3 = np.array([20,10,1])
# arr4 = np.array([[1,2], [3,4]])          # indexing test.
# print('before', arr4)
# arr4.transpose([1,0])       # will not change in place
# print(arr4)
# li1 = [2, 3,5]
# print((arr4>2).sum())   # return a indicating matrix, then sum is the total valid number
# arr_cat = np.vstack([arr_empty, arr4])  # empty  error work .
# arr_map =  map(lambda x:x+1, arr4)  # get map obj
# arr_map = map(lambda x:x+1, li1)  # a
# print(np.array(list(arr_map)))
# arr1 = np.array([1, -1, 2])
# print(arr1.abs()) # no such function check np.ndarray
# arr1 = np.arange(10)        # this is int ,
# print(arr1.dtype)
# arr1 = arr1-0.5     # turn to float
# print(arr1.dtype)
# arr1 = np.array(1)
# print(arr1.flatten()[0])
arr5  = np.array([
    [1,2, 3],
    [4,5,6],
])
arr5_a3 = np.stack((arr5,)*3, axis=-1)

# arr_mask1 = np.array([1, 0])    # can't boradcast,
arr_mask1 = np.array([1, 0, 1]) # only broadcast with last dimension
# print(arr5 * arr_mask1)
# print(arr5>3)       # same shape as arr
# print(arr5_rep)
# print(arr5_a3)
print(arr5_a3[arr5>3])  # indexing with first 2 dim

# print(arr_cat)
# sqr1 = arr4**2
# # print(sqr1)
# R =1    ### MSE PSNR test
# mse1 = (arr4**2).mean()     # for MSE
# mse2 = (arr4[arr4<3]**2).mean()
# # rst_log = 20*np.log(R/arr4**2.mean())
# print(mse1)
# print(mse2)
# print((arr4[arr4>4]<10).sum())  ## empty index? return empty,, sum to 0
## clip test
# arr4.clip(2,3)  # not in place
# print(arr4)
# print(rst_log)
# print(arr2)
# print(arr2/arr3)        # if not same dimension, everything will be expand to the divider size
# arr3[:2,:3] = arr2      # partial index can get all data of 3rd dimension
# print('after assignment arr3 is')
# print(arr3)
# arr1[arr1<-1]=100    # position assigned
# arr1[arr1>2] = 100
# print(arr1)
# check swap
# li1[0], li1[2] = li1[2], li1[0]  # list works fine
# print(li1)
# arr1[:,0], arr1[:,2] = arr1[:,2], arr1[:,0]  # happens in order so the first column is overwritten,  can't work any longer.
# arr1[:,[0,2]] = arr1[:,[2,0]]
# arr1 = arr1.clip(-1,2)
# arr_stk = np.dstack((arr1,arr2))
# print(arr_stk.shape)
# arr_stk = np.dstack((arr_stk, arr3))
# print(arr_stk.shape)
# print(arr_stk)
# print(arr1.sum())
# a = np.random.rand(2,3)

# ############ string operation
# print('{name} is nothing'.format(name=['hey','pig']))
# print('float number {}'.format(100.0))      # can show exactly
# print('{:.3}'.format(128.99816))      # 1.29e+02
# print('{:6.3f}'.format(128.99816))      # 128.998
# print('{:.1f}'.format(128.99816))      # 129.0

## func return test
# def func_rt1():
#     return 1, 2
# x = func_rt1()
# print(x)
# x, _ = func_rt1()
# print(x)

## sum list
# li1 = []
# print(sum(li1))     # 0 returned

# print('haha' + str(0.12) +'.txt')
# print(arr4[:2,:])  # can index only first one with out second colon not always needed

## class test
# class C_test:
#     joints_gt_RGB=1
#     joints_gt_IR = 2
#
# rst = getattr(C_test, 'joints_gt_RGB')
# print(rst)