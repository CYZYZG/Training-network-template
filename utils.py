import numpy as np
import itertools
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import Dataset
import cv2
import os
import math
import torch

def csv_read():
    train_X = []
    test_X = []
    datas = {1: np.array(range(1, 10)), 2: np.array(range(10, 22)), 3: np.array(range(22, 34)),
             4: np.array(range(34, 45)), 5: np.array(range(45, 56)), 6: np.array(range(56, 67)),
             7: np.array(range(67, 79)), 8: np.array(range(79, 87)), 9: np.array(range(87, 97)),
             10: np.array(range(97, 104)), 11: np.array(range(104, 113)), 12: np.array(range(113, 123)),
             13: np.array(range(123, 132)), 14: np.array(range(132, 143)), 15: np.array(range(143, 153)),
             16: np.array(range(153, 159)), 17: np.array(range(159, 168)), 18: np.array(range(168, 178)),
             19: np.array(range(178, 190)), 20: np.array(range(190, 202)), 21: np.array(range(202, 215)),
             22: np.array(range(215, 223)), 23: np.array(range(223, 232)), 24: np.array(range(232, 244)),
             25: np.array(range(244, 256)), 26: np.array(range(256, 265))}
    np.random.seed(46)
    rand_list = np.random.permutation(26) + 1

    for i in range(21):
        for j in datas[rand_list[i]]:
            train_X.append(j)
    train_X = np.array(train_X).reshape((-1, 1))

    for j in range(21, 26):
        for i in datas[rand_list[j]]:
            test_X.append(i)
    test_X = np.array(test_X).reshape((-1, 1))

    return train_X, test_X

class SiameseNetworkDataset(Dataset):

    def __init__(self, Date_X):

        self.Date_X = Date_X
        self.lenss = self.Date_X.shape[0]
        self.dict = {}
        self.target = []
        # 得到每张拼接图像对用于哪个参考图
        for i in os.listdir(r'C:\Users\0001\Desktop\Image Stitching Database\stitched_images'):
            for j in os.listdir(r'C:\Users\0001\Desktop\Image Stitching Database\stitched_images/' + i):
                self.dict[j[:-4]] = i

        for line in open(r"C:\Users\0001\Desktop\Image Stitching Database\MOS.txt", "r"):
            self.target.append(float(line))


    def __getitem__(self, index):
        # 查看参考图的数量
        reg_imglist = os.listdir(r'G:\数据集\图像拼接\Image Stitching Database\constituent_images/' + self.dict[str(self.Date_X[index][0])])
        img_nums = len(reg_imglist)

        img1 = cv2.imdecode(
            np.fromfile(r'G:\数据集\图像拼接\Image Stitching Database\stitched_images/' + str(self.Date_X[index][0])+'.jpg', dtype = np.uint8),
            cv2.IMREAD_GRAYSCALE)
        #img1 = cv2.resize(img1, (7000, 1700))
        img1 = cv2.resize(img1, (2500, 500))

        img1_W = img1.shape[1]
        win_w = math.floor(img1_W / 8)
        temp1 = img1[:, :win_w * 4]
        temp1 = np.expand_dims(temp1, 2)

        temp2 = img1[:, win_w: win_w * 5]
        temp2 = np.expand_dims(temp2, 2)
        temp1 = np.concatenate((temp1, temp2), axis=2)

        temp3 = img1[:, win_w * 3:win_w * 7]
        temp3 = np.expand_dims(temp3, 2)
        temp1 = np.concatenate((temp1, temp3), axis=2)

        temp4 = img1[:,  win_w * 4: win_w * 8]
        temp4 = np.expand_dims(temp4, 2)
        temp1 = np.concatenate((temp1, temp4), axis=2)

        img2 = cv2.imdecode(
            np.fromfile(
                r'G:\数据集\图像拼接\Image Stitching Database\constituent_images/' + self.dict[
                    str(self.Date_X[index, 0])] + '/' + '1.jpg',
                dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img2 = np.empty((int(img2.shape[0]*0.3), int(img2.shape[1]*0.3), 0))

        for img_reg in reg_imglist:
            if img_nums == 5 and img_reg == '3.jpg':
                continue
            temp = cv2.imdecode(
                np.fromfile(
                    r'G:\数据集\图像拼接\Image Stitching Database\constituent_images/' + self.dict[str(self.Date_X[index, 0])]+'/'+ img_reg,
                    dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

            temp = cv2.resize(temp, (int(temp.shape[1]*0.3), int(temp.shape[0]*0.3)))
            temp = np.expand_dims(temp, 2)
            img2 = np.concatenate((img2, temp), axis=2)


        img1 = np.transpose(temp1, (2, 0, 1))
        img1 = torch.from_numpy(img1)
        img1 = torch.tensor(img1, dtype=torch.float)
        #img1 = img1.clone().detach()

        img2 = np.transpose(img2, (2, 0, 1))
        img2 = torch.from_numpy(img2)
        img2 = torch.tensor(img2, dtype=torch.float)
        #img2 = img2.clone().detach()

        score = float(self.target[int(self.Date_X[index, 0])-1])
        score = torch.tensor(score, dtype=torch.float)

        return img1, img2, score

    def __len__(self):
        return self.lenss

# 计算Plcc与Srocc
def computeSpearman(ratings, predictions):
    #predict.data.cpu().numpy()
    a = ratings.data.cpu().numpy()
    b = predictions.data.cpu().numpy()
    sp1 = spearmanr(a, b)
    sp2 = pearsonr(a, b)
    return sp1[0], sp2[0]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count