import torch.utils.data as data
import torch
import h5py
import numpy as np


def norm255(a):
    return a / 255

def norm63(a):
    return a / 63

def norm1024(a):
    return a / 1024
def label_change(label):
    if label == 2:
        label=1
    elif label==3:
        label = 2
    return label


class DatasetFromTXT(data.Dataset):
    def __init__(self,file_path,stare,end):
        super(DatasetFromTXT, self).__init__()
        f = open(file_path, 'r')
        lines = f.readlines()
        ge = (len(lines))

        self.lines = lines
        self.ge = ge
        self.stare = stare
        self.end = end
        f.close()

    def __getitem__(self, index):
        line = self.lines[index]
        List = line.strip('\n').split(' ')
        CUsize = 32
        flagpel = CUsize * CUsize
        flagqp = flagpel + 1
        flagsplit = flagqp + 1

        list1 = list(map(float, List[0: flagpel]))
        list2 = list(map(float, List[flagpel: flagqp]))
        list3 = list(map(float, List[flagqp: flagsplit]))


        list1 = list(map(norm1024, list1))
        list2 = list(map(norm63, list2))

        Pel = np.zeros((1, flagpel), 'float32')
        Qp = np.zeros((1, flagqp - flagpel), 'float32')
        label = np.zeros((1, 1), 'float32')
        Pel[0, :] = list1
        Qp[0, :] = list2
        label[0, :] = list3

        Pel = Pel.reshape(1, CUsize, CUsize)

        return  Pel, Qp, label




    def __len__(self):
        #return self.data.shape[0]
        return self.ge