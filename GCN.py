import torch.utils.data as data
import torch
import h5py
import numpy as np

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path,'r',libver='latest',swmr=True)
        self.data = hf.get('data')
        self.target = hf.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:]).float(), torch.from_numpy(self.target[index,:]).float()

    def __len__(self):
        return self.data.shape[0]

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

        #data = np.zeros((ge, end - stare + 1), dtype=float)
        #label = np.zeros((ge,1),dtype=float)
        #cnt = 0
        #for line in lines:
        #    List = line.strip('\n').split(' ')
        #    list1 = list(map(float, List[(stare - 1):end]))
        #    list2 = list(map(float,List[end:end + 1]))
        #    if stare == 1:
        #        list1 = list(map(norm, list1))  # ZJQ 归一化到0-255之间
        #    data[cnt, :] = list1
        #    label[cnt, :] = list2
        #    cnt += 1
        #self.data = data.reshape((cnt - 1, 9, 32, 32))
        #self.label = label
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

        # narray = np.array(list1)
        # sum1 = narray.sum()
        # n1 = len(narray)
        # mean1 = sum1/n1
        list1 = list(map(norm1024, list1))
        list2 = list(map(norm63, list2))
        # list3 = list(map(label_change, list3))

        Pel = np.zeros((1, flagpel), 'float32')
        Qp = np.zeros((1, flagqp - flagpel), 'float32')
        label = np.zeros((1, 1), 'float32')
        Pel[0, :] = list1
        Qp[0, :] = list2
        label[0, :] = list3

        Pel = Pel.reshape(1, CUsize, CUsize)

        return  Pel, Qp, label


        #return torch.from_numpy(self.data[index, :]).float(), torch.from_numpy(self.target[index, :]).float()

    def __len__(self):
        #return self.data.shape[0]
        return self.ge