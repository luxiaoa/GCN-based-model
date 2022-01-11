# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import warnings
warnings.filterwarnings('ignore')
import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import  FCN_GCN
from dataset import DatasetFromHdf5
from dataset import DatasetFromTXT
from dataloaderX import DataloaderX
from torch.utils.tensorboard import SummaryWriter
from util import FocalLoss, SoftDiceLoss


parser = argparse.ArgumentParser(description="PyTorch VVC_ml")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=1000, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=40, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda?")
parser.add_argument("--resume", default='', type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=4, help="Number of threads for data loader to use, Default: 1")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--num_classes", type=int, default=6, help="classes number out")
parser.add_argument("--focal_gamma", type=int, default=2, help="focal loss")
parser.add_argument("--train_num", type=int, default=4.2, help="train num flag")
parser.add_argument("--train_path", type=str, default="", help="train dataset path")

def main():
    global opt, model, focalLoss, Diceloss
    opt = parser.parse_args()
    print(opt)

    tensorboard_runs = 'train/tensorboard'
    train_writer = SummaryWriter(os.path.join(tensorboard_runs, 'train'))

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True



    print("===> Building model")
    model = FCN_GCN(num_classes=1)
    focalLoss = FocalLoss(class_num=6, alpha=None, gamma=opt.focal_gamma, size_average=True)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        focalLoss = focalLoss.cuda()

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    fileindex = [n for n in range(0, 48)]
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        random.shuffle(fileindex)
        for filenum in fileindex:
            print("===> Loading datasets")
            train_set = DatasetFromTXT(opt.train_path + str(filenum) + ".txt", 1, 9216)
            training_data_loader = DataloaderX(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                          shuffle=True)
            train(training_data_loader, optimizer, model,focalLoss,epoch, filenum, train_writer)
        save_checkpoint(model, epoch)


pointnum = 0
def train(training_data_loader, optimizer, model, criterion,  epoch, filenum, train_writer):#ZJQ 训练集，优化器，网络模型，损失函数，代
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        inp, QP, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)

        target = torch.squeeze(torch.squeeze(target, 1), 1)
        if opt.cuda:
            inp = inp.cuda()
            QP = QP.cuda()
            target = target.cuda()
        model_input = inp
        out = model(model_input, QP)
        if(target.shape[0] == 0):
            print("out error")
        loss_focal = focalLoss(out, target.long())
        loss = loss_focal

        loss_total = {'loss_criter': loss}
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if iteration%100 == 0:
            print("===> Epoch[{}]({}/{})-{}: Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), filenum,
                                                                loss_total['loss_criter']))
            global pointnum
            train_writer.add_scalar('train/Loss_criter_train'+ str(opt.train_num) , loss_total['loss_criter'], pointnum)
            pointnum += 1

def save_checkpoint(model, epoch):
    model_out_path = "checkpoint_train"+ str(opt.train_num) + "/model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint_train"+ str(opt.train_num)):
        os.makedirs("checkpoint_train"+ str(opt.train_num))

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))
if __name__ == '__main__':
    main()

