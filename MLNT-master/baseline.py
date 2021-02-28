from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets

import torchvision
import torchvision.transforms as transforms
import models as models

import os
import sys
import time
import argparse
import datetime

from torch.autograd import Variable
import dataloader_cifar
import PreResNet

import dataloader

parser = argparse.ArgumentParser(description='PyTorch Clothing-1M Training')
parser.add_argument('--lr', default=0.0008, type=float, help='learning_rate')
parser.add_argument('--start_epoch', default=2, type=int)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--optim_type', default='SGD')
parser.add_argument('--seed', default=7)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--id', default='cross_entropy')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset: cifar10 / cifar100')
parser.add_argument('--perturb_ratio', default=0.5, type=float, help='ratio of random perturbations')
parser.add_argument('--noise_rate', default=0.4, type= float)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
  
# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    learning_rate = args.lr
    if epoch > args.start_epoch:
        learning_rate=learning_rate/10        
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    i = 0
    print('\n=> %s Training Epoch #%d, LR=%.4f' %(args.id,epoch, learning_rate))
    #print(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print("start one batch ", batch_idx)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        #print('line 67')
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs) # Forward Propagation
        #print('input shape is ', inputs.shape)
        #print('output shape is ', outputs.shape)
        #print('target shape is ', targets.shape)
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update
        #print('line 73')
        #print(loss.data[0])
        #train_loss += loss.data[0]
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        #loss.data[0]
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, loss.item(), 100.*correct/total))
        #print('line 90')
        sys.stdout.flush()
        #print('line 92')
        if batch_idx%1000==0 :
            #print(batch_idx%1000==0)
            print('line 94')
            val(epoch)
            print('line 97')
            net.train()
        print('train end for one batch')

            
def val(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        #test_loss += loss.data[0]
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    #loss.data[0]
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))
    record.write('Validation Acc: %f\n'%acc)
    record.flush()
    if acc > best_acc:
        best_acc = acc
        print('| Saving Best Model ...')
        save_point = './checkpoint/%s.pth.tar'%(args.id)
        save_checkpoint({
            'state_dict': net.state_dict(),
        }, save_point) 

def test():
    global test_acc
    test_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = test_net(inputs)
        loss = criterion(outputs, targets)

        #test_loss += loss.data[0]
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    acc = 100.*correct/total   
    test_acc = acc
    record.write('Test Acc: %f\n'%acc)

if os.path.exists('checkpoint'):
    pass
else:
    os.mkdir('checkpoint')

record=open('./checkpoint/'+args.id+'_test.txt','w')
record.write('learning rate: %f\n'%args.lr)
record.flush()

'''    
loader = dataloader.clothing_dataloader(batch_size=args.batch_size,num_workers=5,shuffle=True)
train_loader,val_loader,test_loader = loader.run()
'''
if args.dataset=='cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True)
else:
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)

loader = dataloader_cifar.cifar_dataloader(args.dataset,r=args.noise_rate,noise_mode='instance',batch_size=args.batch_size,num_workers=5,\
    root_dir='./')

val_loader = loader.run('eval_train')
test_loader = loader.run('test')
train_loader = loader.run('warmup')
###
#addded
print('data already loaded')

best_acc = 0
test_acc = 0
# Model
print('\nModel setup')
print('| Building net')
#net = models.resnet50(pretrained=True)
#net = models.resnet18(pretrained=True)
if args.dataset=='cifar10':
    num_class = 10
elif args.dataset == 'cifar100' :
    num_class = 100
else:
    num_class = 14 # clthing dataset
net = PreResNet.ResNet18(num_class)
net.fc = nn.Linear(2048,num_class)
#test_net = models.resnet50(pretrained=True)
test_net = PreResNet.ResNet18(num_class)
test_net.fc = nn.Linear(2048,num_class)
if use_cuda:
    net.cuda()
    test_net.cuda()
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

print('\nTraining model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(args.optim_type))

for epoch in range(1, 1+args.num_epochs):
    train(epoch)
    val(epoch)

print('\nTesting model')
checkpoint = torch.load('./checkpoint/%s.pth.tar'%args.id)
test_net.load_state_dict(checkpoint['state_dict'])
test()

print('* Test results : Acc@1 = %.2f%%' %(test_acc))
record.write('Test Acc: %.2f\n' %test_acc)
record.flush()
record.close()
