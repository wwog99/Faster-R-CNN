import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2

from model import Faster_rcnn
from dataset import Image_loader

def train(model, device, train_loader, epoch, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        input, label = data
        label = torch.Tensor(label)
        print(input.shape)
        print(label.shape)
        input, label = Variable(input), Variable(label)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward
        optimizer.step()
        running_loss += loss.data[0]
        print("{}th epoch running loss : {}".format(epoch, running_loss))


def test(model, device, test_loader, epoch):
    model.eval()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = Image_loader()
    test_loader = Image_loader()
    model = Faster_rcnn()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    epochs =1
    for epoch in range(1, epochs+1):
        train(model, device, train_loader, epoch, optimizer,criterion)
    test(model, device, test_loader, epoch)

if __name__ == '__main__':
    main()
