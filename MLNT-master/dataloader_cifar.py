from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
import argparse
#import modules.data_utils as datasets
import sys


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='',clean_file='', pred=[], probability=[], log=''):
        
        self.r = r # noise ratio
        self.transform = transform
        self.noise_mode = noise_mode
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
     
        if self.mode=='test':
            if dataset=='cifar10':
                test_dic = unpickle('%s/data/cifar-10-batches-py/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/data/cifar-100-python/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10':
                #print("current path is {}".format(sys.path[0]))
                for n in range(1,6):
                    dpath = '%s/data/cifar-10-batches-py/data_batch_%d'%(root_dir,n)
                    #print("path is {}".format(dpath))
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/data/cifar-100-python/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            train_label = np.array(train_label)
            noise_label = train_label.copy()
            if dataset == 'cifar10':
                nb_classes = 10
            elif dataset == 'cifar100':
                nb_classes = 100
            clean_per_class = int(5000 / nb_classes) # cifar10: 100 else: 10
            noise_per_class = int(50000/nb_classes*r)


            #select clean_per_class numbers of data in each class as clean data
            #leave the other data to add noise
            #the 0th data processing is at the outer loop
            #0th add noise (for index)
            all_index = np.arange(50000).reshape(-1)
            clean_indices = all_index[np.where(train_label == 0)[0]][-clean_per_class:]
            noise_idx = [all_index[np.where(train_label == 0)[0]][:-clean_per_class]]
            #from 1th to 9th to add noise (for index)
            for i in range(nb_classes-1):
                indices1 = all_index[np.where(train_label == i+1)[0]][-clean_per_class:]
                noisy_indices1 = all_index[np.where(train_label == i+1)[0]][:-clean_per_class]
                clean_indices = np.concatenate((clean_indices, indices1))
                noise_idx.append(noisy_indices1)
            #add noise
            for t,i in enumerate(noise_idx):
                # randomly selected one image as the center
                image_center = train_data[i[10]]
                norm_loss = np.zeros(len(i))
                for j,k in enumerate(i):
                    images = train_data[k]
                    norm_loss[j] = np.linalg.norm(image_center - images)
                noisy_indices = i[norm_loss.argsort()[:noise_per_class]]
                noise_label[noisy_indices] = (t+1)%nb_classes


            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            elif self.mode == 'small':
                self.train_data = train_data[::100]
                self.noise_label = noise_label[::100]
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    #clean = (np.array(noise_label)==np.array(train_label))
                    clean = (noise_label == train_label)
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = noise_label[pred_idx]
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            #img = Image.fromarray(img)
            img = Image.fromarray(img).convert('RGB')
            img = self.transform(img)
            #print(img.size())
            #print(target.size())
            return img, target
        elif self.mode=='small':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            #img = Image.fromarray(img)
            img = Image.fromarray(img).convert('RGB')
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log= '', noise_file='',clean_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.clean_file = clean_file
        if self.dataset=='cifar10':
            '''
            self.transform_train = transforms.Compose([
                    transforms.Resize(256),
                # transforms.RandomSizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
             ])  # meanstd transformation

            self.transform_test = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            '''
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    

        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])


    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file, clean_file = self.clean_file)
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader

        elif mode=='small':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="small",noise_file=self.noise_file, clean_file = self.clean_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, clean_file = self.clean_file, pred=pred, probability=prob,log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, clean_file = self.clean_file, pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file, clean_file = self.clean_file)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--batch_size', default=256, type=int, help='train batchsize')
    parser.add_argument('--noise_mode',  default='instance')
    parser.add_argument('--r', default=0.3, type=float, help='noise ratio')

    parser.add_argument('--id', default='')
    parser.add_argument('--seed', default=123)
    parser.add_argument('--gpuid', default=0, type=int)

    parser.add_argument('--num_class', default=10, type=int)
    parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', type=str)

    args = parser.parse_args()

    #torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #stats_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w')
    #test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')

    loader = cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path)

    val_loader = loader.run('eval_train')
    test_loader = loader.run('test')
    train_loader = loader.run('warmup')
    small_loader = loader.run('small')
    print(len(small_loader.dataset))





