from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import time

from sklearn.metrics import confusion_matrix

from torchvision import datasets, transforms

import itertools

gound_truth_list = []
answer_list = []
# total_epoch = 1000
# Leaning_Rate = 0.001

model_type ="mymodel"
#model_type ="VGG"


#실험용 confusion matrix 
total_epoch = 20
Leaning_Rate = 0.01

def imshow(inp, cmap=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap)

class Net(nn.Module):
    # def __init__(self, output_size):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
    #     self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #     self.conv2_drop = nn.Dropout2d()
    #     self.fc1 = nn.Linear(500, 120, bias=True)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

    #     x = x.view(x.size(0), -1)

    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = F.relu(self.fc3(x))

    #     return F.log_softmax(x, -1)

    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=2)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = self.drop1(x)
        
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = self.drop1(x)

        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        x = self.drop1(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)

        x = F.relu(self.fc2(x))
        return F.log_softmax(x, -1)

def fit(epoch, model, data_loader, phase='training', volatile=False, is_cuda=True):
    if model_type == "mymodel":
        optimizer = optim.SGD(model.parameters(), lr=Leaning_Rate, momentum=0.5)

    elif model_type == "VGG":
        optimizer = optim.SGD(model.classifier.parameters(), lr=Leaning_Rate, momentum=0.5)

    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True

    running_loss = 0.0
    running_correct = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data = data.to('cuda')
            target = target.to('cuda')
        else:
            data = data.to('cpu')
            target = target.to('cpu')

        if phase == 'training':
            optimizer.zero_grad()

        output = model(data)

        if model_type == "mymodel":
            loss = F.nll_loss(output, target)
            running_loss += F.nll_loss(output, target, size_average=False).data
        elif model_type == "VGG":
            loss = F.cross_entropy(output, target)
            running_loss += F.cross_entropy(output, target, size_average=False).data

        preds = output.data.max(dim=1, keepdim=True)[1]

        gound_truth = target.data

        answer = preds.squeeze()

        a = gound_truth.data.detach().cpu().numpy()
        b = answer.data.detach().cpu().numpy()

        gound_truth_list.append(a)
        answer_list.append(b)

        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct.item() / len(data_loader.dataset)
    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')

    return loss, accuracy

def training():
    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True

        print("cuda support")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("cpu support")
        torch.set_default_tensor_type('torch.FloatTensor')

    TRAIN_PATH = "./data/train"
    TEST_PATH = "./data/test"

    simple_transform = transforms.Compose([transforms.Resize((96, 96))
                                              , transforms.ToTensor()
                                              , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])
    train = ImageFolder(TRAIN_PATH, simple_transform)
    test = ImageFolder(TEST_PATH, simple_transform)
    
    print("len data1:{}".format(len(train)))
    print("len data2:{}".format(len(test)))

    print("class2idx:{}".format(train.class_to_idx))
    print("class:{}".format(train.classes))
    print("len:{}".format(len(train.classes)))

    train_data_loader = torch.utils.data.DataLoader(train, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(test, batch_size=64, num_workers=4, shuffle=False)

    print("------------- data load finished -------------------------")

    if model_type == "mymodel":
        model = Net(len(train.classes))

        if is_cuda:
            model.cuda()
            print("model send to cuda()")

    elif model_type == "VGG":
        #VGG 모델
        model = models.vgg16(pretrained=False)
        model = model
        print("model:{}".format(model) )
        model.classifier[6].out_features = 9

    graph_epoch = []
    train_losses = []
    train_accuracy = []
    val_losses = []
    val_accuracy = []

    print("is_cuda:{}".format(is_cuda))

    for epoch in range(1, total_epoch):
        print("-----------training: {} epoch-----------".format(epoch))

        epoch_loss, epoch_accuracy = fit(epoch, model, train_data_loader, phase='training')
        val_epoch_loss, val_epoch_accuracy = fit(epoch, model, valid_data_loader, phase='validation')

        graph_epoch.append(epoch)

        a = epoch_loss.detach().cpu().data.item()
        b = epoch_accuracy
        c = val_epoch_loss.detach().cpu().data.numpy()
        d = val_epoch_accuracy

        train_losses.append(a)
        train_accuracy.append(b)
        val_losses.append(c)
        val_accuracy.append(d)

        print("train_losses:{}".format(train_losses))

        if epoch % 10 == 1:
            savePath = "./model/model_" + str(model_type) + str(epoch) + ".pth"
            torch.save(model.state_dict(), savePath)
            print("file save at {}".format(savePath))

    x_len = np.arange(len(train_losses))
    plt.plot(x_len, train_losses, marker='.', lw =1, c='red', label="train_losses")
    plt.plot(x_len, val_losses, marker='.', lw =1, c='cyan', label="val_losses")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.plot(x_len, val_accuracy, marker='.', lw =1, c='green', label="val_accuracy")
    plt.plot(x_len, train_accuracy, marker='.', lw =1, c='blue', label="train_accuracy")

    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

    gound_truth_list_1 = []
    for idx, data in enumerate(gound_truth_list):
        for j in data:
            gound_truth_list_1.append(j)

    # print("gound truth list1:{}".format(gound_truth_list_1))

    ans_truth_list_1 = []
    for idx, data in enumerate(answer_list):
        for j in data:
            ans_truth_list_1.append(j)

    # print("ans list1:{}".format(ans_truth_list_1))

    plt.show()

if __name__ == '__main__':
    training()