from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import data
import utils
import util
import torch.nn as nn
import torch.optim as optim
import models
from preprocess import get_transform
import time
import logging
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import models
import matplotlib.pyplot as plt
from data import get_dataset
from preprocess import get_transform
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
from utils import *
import math
import numpy
from sklearn.decomposition import PCA
os.environ['CUDA_VISIBLE_DEVICES']='0'

#from models import resnet
from torch.autograd import Variable
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
#print('printing models dict %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#for name in models.__dict__ :
#  print(name)


#print('printing model names %%%%%%%%%%%%%%%%%%%%%%%%%%')
#print(model_names)
def run_PCA(activations_collect,key_idx, components, threshold=0.999):
        """threshold for minimal loss in performance=0.999
        activations_collect  function gathers activations over enough mini batches.
        components=number of filters in the layer you are compressing
        This is for a layer, you need to run this for multiple layers and store optimal_num_filters into a vector
        This vector is the significant dimensionality of all layers"""
        
        print('number of components are',components)
        activations=activations_collect[key_idx]#.replace('.weight','')]
	activations = (activations.data).cpu().numpy()
#	numpy.savetxt('./Act_files_'+str(key_idx)+'.out',activations)
        print('shape of activations are:',activations.shape)
        a=activations.swapaxes(1,2).swapaxes(2,3)
        a_shape=a.shape
        print('reshaped ativations are of shape',a.shape)
	raw_input()
#	numpy.savetxt('./Act_files_'+str(key_idx)+'.out',a.reshape(a_shape[0]*a_shape[1]*a_shape[2],a_shape[3]))

        pca = PCA(n_components=components) #number of components should be equal to the number of filters
        pca.fit(a.reshape(a_shape[0]*a_shape[1]*a_shape[2],a_shape[3])) #this should be N*H*W,M
        a_trans=pca.transform(a.reshape(a_shape[0]*a_shape[1]*a_shape[2],a_shape[3]))
        print('explained variance ratio is:',pca.explained_variance_ratio_)
	raw_input()
        plt.plot(numpy.cumsum(pca.explained_variance_ratio_))
	numpy.savetxt('./PCA_files_'+str(key_idx)+'.out',numpy.cumsum(pca.explained_variance_ratio_))
#        plt.savefig('./PCA_files_'+str(key_idx)+'.jpeg')  #saves the PCA figure.
	raw_input()
        optimal_num_filters=numpy.sum(numpy.cumsum(pca.explained_variance_ratio_)<threshold) 
        print('we want to retain this percentage of explained variance',threshold)
        print('number of filters required to explain that much variance is',optimal_num_filters)
	return optimal_num_filters,pca.components_
#	return a_shape
def accuracy(output, target, training, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #print(target)
    #print(output)
    #print(target.view(1, -1).expand_as(pred))
    #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    if training:
    	correct = pred.eq(target.data.view(1, -1).expand_as(pred))
    else:
	correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
class AverageMeter(object):
    """Computes and stores the average and current value"""
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
def save_state(model, best_acc):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'models/resnet20xnorinf3.pth.tar')

def train(epoch,model):
    global best_train_acc
    model.train()
    training = True
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    correct = 0
    for batch_idx, (inputs, target) in enumerate(trainloader):
        # process the weights including binarization
        bin_op.binarization()
        if args.gpus is not None:
            target = target.cuda(async=True)
        data = Variable(inputs.type(args.type), volatile=not training)
        target = Variable(target)
        
        # forwarding
       # data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        #print(model)
	#print(data)
	output = model(data)
        
        # backwarding
	#print(criterion)
	#print(output)
	#print(target)
        loss = criterion(output, target)
        loss.backward()
        
        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        
        optimizer.step()
	prec1, prec5 = accuracy(output.data, target, training, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        #pred = output.data.max(1, keepdim=True)[1]
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    	
    	#save_state(model, best_train,acc)
        if batch_idx % 10 == 0:
	    print('Epoch: [{0}][{1}/{2}({3:.0f}%)]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\tLR: {LR}'.format(
                   epoch, batch_idx, len(trainloader), 100. *float(batch_idx)/len(trainloader),
                   loss=losses, top1=top1, top5=top5, LR = optimizer.param_groups[0]['lr']))

    acc = top1.avg
    if acc > best_train_acc:
        	best_train_acc = acc
    print('Train Epoch: {}\t({:.2f}%)]\tLoss: {:.6f}\n'.format(
                epoch,
                acc, losses.avg))

    print('Best Accuracy: {:.2f}%\n'.format(best_train_acc))
    return acc, losses.avg
def test():
    global best_acc
    model.eval()
    flag=True
    training=False
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    bin_op.binarization()
    for batch_idx,(data, target) in enumerate(testloader):
        target = target.cuda(async=True)
        data_var = torch.autograd.Variable(data.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        
	output,activations = model(data_var)
	#Layers to run PCA on
	key_idx = range(0,19)
	
	for i in key_idx: #Run PCA layer-wise
            size_keyidx = activations[i].size()
	    activation_i = activations[i]
	    run_PCA(activations,i,size_keyidx[1], threshold=0.99)
	
	
	loss= criterion(output, target_var)

	prec1, prec5 = accuracy(output.data, target, training, topk=(1, 5))
        losses.update(loss.data[0], data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

	if flag == True:
	     if batch_idx % 10 == 0:
	       print('[{0}/{1}({2:.0f}%)]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   batch_idx, len(testloader), 100. *float(batch_idx)/len(testloader),
                   loss=losses, top1=top1, top5=top5))
	else:
	    if batch_idx % 10 == 0:
	       print('Epoch: [{0}][{1}/{2}({3:.0f}%)]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, batch_idx, len(testloader), 100. *float(batch_idx)/len(testloader),
                   loss=losses, top1=top1, top5=top5))
    bin_op.restore()
    acc = top1.avg
    if acc > best_acc:
        best_acc = acc
	if flag == False:
           save_state(model, best_acc)
    #test_loss /= len(testloader.dataset)
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.4f}'
          .format(top1=top1, top5=top5,loss = losses))

    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return acc, losses.avg

   
def adjust_learning_rate(optimizer, epoch):
    update_list = [81, 122, 160, 240]
    lr_list = [1e-3,5e-4,1e-4,1e-5]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
	    #print(epoch)
	    #raw_input()
            index_epoch = update_list.index(epoch)
	    #print(index_epoch)
	    #raw_input()
	    param_group['lr'] = lr_list[index_epoch]
    return

if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='/data2/ichakra/BinaryNet.pytorch/Datasets/CIFAR100/cifar-100-python',
            help='dataset path')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
                    help='dataset name or folder')
    parser.add_argument('--arch', action='store', default='resnet20hybunrolled',
            help='the architecture for the network: resnet')
    parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20hybunrolled',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet)')

    parser.add_argument('--lr', action='store', default='0.1',
            help='the intial learning rate')
    parser.add_argument('--momentum', action='store', default='0.9',
            help='momentum')
    parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')

    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
		                        
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')

    parser.add_argument('--gpus', default='2',
                    help='gpus used for training - e.g 0,1,3')
    parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
    parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
    parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')

    args = parser.parse_args()
    print('==> Options:',args)
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'txt', results_file % 'html')

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    args.evaluate = True 
   
    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'resnethyb1' or 'resnet' or 'resnethyb' or 'resnethybunrolled':  
        #print(models.__dict__)
        model = models.__dict__[args.model]
	model_config = {'input_size': args.input_size, 'dataset': args.dataset}
        print(model,model_config)
    else:
        raise Exception(args.arch+' is currently not supported')
    
    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    print(model)

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
	best_train_acc=0
        for m in model.modules():
	  #  print (m)for m in model.modules():
	  if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
          elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])
	

    if not args.cpu:
        model.cuda()
	if args.gpus and len(args.gpus) > 1:
        	model = torch.nn.DataParallel(model, args.gpus)

        #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)
     # prepare the data
    if not os.path.isfile(args.data+'/train'):
        # check the data path
        raise Exception\
                ('Please assign the correct data path with --data <DATA_PATH>')
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    train_data = get_dataset(args.dataset, 'train', transform['train'])
    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)


    #trainset = data.dataset(root=args.data, train=True)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
           # shuffle=True, num_workers=2)
    test_data = get_dataset(args.dataset, 'val', transform['eval'])
    testloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

   # testset = data.dataset(root=args.data, train=False)
   # testloader = torch.utils.data.DataLoader(testset, batch_size=100,
          #  shuffle=False, num_workers=2)

    # define classes
    classes = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100)

    # define solver and criterion
    base_lr = float(args.lr)
    
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr, 
            'weight_decay':0.00001}]

        optimizer = optim.SGD(params, momentum = float(args.momentum), lr=0.10,weight_decay=0.00001, nesterov=True,dampening=0)
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.evaluate:
        test()
        exit(0)

    # start training
    for epoch in range(1, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        [trainacc,train_loss] = train(epoch,model)
        [testacc,test_loss] = test()
	results.add(epoch=epoch, train_loss=train_loss, test_loss=test_loss,
                    train_error1=100 - trainacc, test_error1=100 - testacc)
        results.save()
        plt.ion()
        fig1 = plt.figure(1)
    	ax1 = fig1.add_subplot(1,1,1)
    	ax1.plot(epoch,train_loss,'rs-',epoch,test_loss,'bs-')
    	plt.ylabel('loss')
    	fig1.canvas.draw()
    	fig2 = plt.figure(2)
    	ax2 = fig2.add_subplot(1,1,1)
    	ax2.plot(epoch,100-trainacc,'rs-',epoch,100-testacc,'bs-')
    	plt.ylabel('error %')
    	fig2.canvas.draw()
