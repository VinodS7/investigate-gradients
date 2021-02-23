from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR
import numpy as np

import matplotlib.pyplot as plt


import attribution_methods
from  model import Net

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss +=F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_raw_data(args, model, device, test_data, test_label):
    
    model.eval()
    test_loss = 0
    correct = 0
    idx = np.arange(len(test_data))
    with torch.no_grad():
        for i in range(0, len(test_data), args.test_batch_size):
            if(i+args.test_batch_size>len(test_data)):
                temp = idx[i:]
            else:
                temp = idx[i:i+args.test_batch_size]
            batch_data = test_data[temp,:,:,:]
            batch_label = test_label[temp]
            output = model(torch.from_numpy(batch_data).to(device))
            if(args.classifier_type=='softmax'):
                test_loss+= F.nll_loss(output, torch.from_numpy(batch_label).to(device))
                pred = output.argmax(dim=1, keepdim=True)
                temp = pred.eq(torch.from_numpy(batch_label).to(device).view_as(pred)).numpy() 
                temp = i+np.where(temp==False)[0]
                if temp.size>0:
                    input(temp)
            
                correct += pred.eq(torch.from_numpy(batch_label).to(device).view_as(pred)).sum().item()
            elif(args.classifier_type=='sigmoid'):
                test_loss+= F.binary_cross_entropy(torch.squeeze(output), torch.from_numpy(batch_label.astype(np.float32)).to(device))
                pred = torch.round(output)
                pred = pred.type(torch.int64)
                correct += pred.eq(torch.from_numpy(batch_label).to(device).view_as(pred)).sum().item()
   
        test_loss /= len(test_data)
        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data),
        100. * correct / len(test_data)))

def loss_grad(model_fn, x, label, classifier_type):
    x = x.clone().detach().requires_grad_(True)
    #x = x.requires_grad_()
    if(classifier_type=='softmax'):
        loss = F.nll_loss(model_fn(x), label)
    elif(classifier_type=='sigmoid'):
        loss = F.binary_cross_entropy(torch.squeeze(model_fn(x)), label.type(torch.float32))
 
    grad, = torch.autograd.grad(loss, [x])
    return grad.cpu().detach().numpy()

def saliency_map(model_fn, x, label, classifier_type):
    x = x.clone().detach().requires_grad_(True)
    if(classifier_type=='softmax'):
        outputs = model_fn(x)
        #input(outputs)
        outputs_max, _ = torch.max(outputs,dim=1)
        #input(outputs_max.size())
        #outputs_max = outputs[:,outputs_max_index]
        #input(outputs_max.size())
        grad, = torch.autograd.grad(outputs_max, [x])
    elif(classifier_type=='sigmoid'):
        outputs = model_fn(x)
        grad, = torch.autograd.grad(outputs, [x])


    #input(grad.size())
    return grad.cpu().detach().numpy()




       
def test_attributed_data(args, model, device, test_data, test_label, attribution_method):

    model.eval()
    test_loss = 0
    correct = 0
    idx = np.arange(len(test_data))
    for i in range(0, len(test_data), args.test_batch_size):
        if(i+args.test_batch_size>len(test_data)):
            temp = idx[i:]
        else:
            temp = idx[i:i+args.test_batch_size]
        batch_data = test_data[temp,:,:,:]
        batch_label = test_label[temp]  
        g = loss_grad(model, torch.from_numpy(batch_data).to(device).requires_grad_(True), torch.from_numpy(batch_label).to(device), args.classifier_type)
        #g = saliency_map(model, torch.from_numpy(batch_data).to(device).requires_grad_(True), torch.from_numpy(batch_label).to(device), args.classifier_type)
        #print(g.shape)
        #input(s.shape)
        for j in range(batch_data.shape[0]):
            #attribution_method.contribution_signs(batch_data[j,:,:,:].copy(), g[j])
            batch_data[j,:,:,:], v = attribution_method.apply_attribution(batch_data[j,:,:,:].copy(), g[j])
        output = model(torch.from_numpy(batch_data).to(device))
        if(args.classifier_type=='softmax'):
            test_loss+= F.nll_loss(output, torch.from_numpy(batch_label).to(device))
            pred = output.argmax(dim=1, keepdim=True)
            temp = pred.eq(torch.from_numpy(batch_label).to(device).view_as(pred)).numpy() 
            temp = i+np.where(temp==False)[0]
            #if temp.size>0:
            #    input(temp)
            correct += pred.eq(torch.from_numpy(batch_label).to(device).view_as(pred)).sum().item()
        elif(args.classifier_type=='sigmoid'):
            test_loss+= F.binary_cross_entropy(torch.squeeze(output), torch.from_numpy(batch_label.astype(np.float32)).to(device))
            pred = torch.round(output)
            pred = pred.type(torch.int64)
            correct += pred.eq(torch.from_numpy(batch_label).to(device).view_as(pred)).sum().item()
    #print('positive inputs: ',attribution_method.ctrb_sgns[0,:])
    #print('negative inputs: ', attribution_method.ctrb_sgns[1,:])
    #print('zero inputs: ', attribution_method.ctrb_sgns[2,:])
    #input(attribution_method.ctrb_sgns/np.sum(attribution_method.ctrb_sgns))
    test_loss /= len(test_data)
        
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data),
        100. * correct / len(test_data)))
    return correct/len(test_data)

def plot_images(args, model, device, test_data, test_label, attribution_method, num=10, indices=None):
    model.eval()
    if indices is not None:
        data_len = len(indices)
        idx = indices
    else:
        data_len = num
        idx = np.arange(len(test_data))
        np.random.shuffle(idx)
    
    for i in range(0, data_len, args.test_batch_size):
        if(i+args.test_batch_size>data_len):
            temp = idx[i:data_len]
        else:
            temp = idx[i:i+args.test_batch_size]
        
        batch_data = test_data[temp,:,:,:]
        batch_label = test_label[temp]  
        g = loss_grad(model, torch.from_numpy(batch_data).to(device).requires_grad_(True), torch.from_numpy(batch_label).to(device), args.classifier_type)
 
        for j in range(batch_data.shape[0]):
            temp = (batch_data[j]*0.3081)+0.1307
            temp = np.squeeze(temp)
            plt.imshow(temp, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            plt.axis('off')
            plt.savefig('images/mnist1_inv/orig_img_'+str(idx[i+j])+'.png',bbox_inches='tight', pad_inches=0)
            plt.close()
            batch_data[j,:,:,:], v = attribution_method.apply_attribution(batch_data[j,:,:,:].copy(), g[j])
            #attribution_method.contribution_signs(batch_data[j,:,:,:].copy(), g[j])
            temp = (batch_data[j]*0.3081)+0.1307
            temp = np.squeeze(temp)
            plt.imshow(temp, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            plt.axis('off')
            plt.savefig('images/mnist1_inv/mod_img_'+str(idx[i+j])+args.attribution_method+'.png', bbox_inches='tight', pad_inches=0)
            plt.close()

        output = model(torch.from_numpy(batch_data).to(device))
        print(output.argmax(dim=1, keepdim=True))

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
            help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
            help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
            help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
            help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
            help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry_run', action='store_true', default=False,
            help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
            help='For saving current model')
    parser.add_argument('--num-classes', type=int, default=10,
            help='number of classes for classification')
    parser.add_argument('--ROAR', type=int, default=1,
            help='Flag to determine KAR or ROAR')
    parser.add_argument('--attribution-method', type=str, default='random',
            help='attribution method for occlusion')
    parser.add_argument('--occlude', type=int, default=0,
            help='number of pixels to remove')
    parser.add_argument('--classifier-type', type=str, default='softmax',
            help='choose between softmax and singular sigmoid output'
            'sigmoid output only compatible with 2 class case')
    parser.add_argument('--replacement-type', type=str, default='mean',
            help='Value to replace the occluded part of input by')
    parser.add_argument('--replacement-value', type=float, default=0,
            help='if replacement-type is custom use this value')
    parser.add_argument('--model-path', nargs='+', type=str, default=None,
            help='Provide path and filename to save model')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_kwargs = {'batch_size':args.batch_size}
    test_kwargs = {'batch_size':args.test_batch_size}

    
    cuda_kwargs = {'num_workers': 1,
                'pin_memory': True,
                'shuffle': True}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
            transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
            transform=transform)
    
    #attribution_method = attribution_methods.AttributionMethods(args.ROAR, args.occlude, 28, args.attribution_method,
    #        replacement=args.replacement_type, dataset_min=-0.1307/0.3081, custom_value=args.replacement_value) 
    #train_data = dataset1.data.numpy()
    #test_data = dataset2.data.numpy()
    train_label = dataset1.targets.numpy()
    test_label = dataset2.targets.numpy()
    count = np.zeros(10)
    idx_train = []
    idx_test = []
    for i in range(len(train_label)):
        count[train_label[i]]+=1
        if(train_label[i]<args.num_classes):
            idx_train.append(i)
    for i in range(len(test_label)):
        count[train_label[i]]+=1
        if(test_label[i]<args.num_classes):
            idx_test.append(i)
    

    
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    train_load = torch.utils.data.DataLoader(dataset1, batch_size=len(dataset1))
    test_load = torch.utils.data.DataLoader(dataset2, batch_size=len(dataset2))
    
    train_data = next(iter(train_load))[0].numpy()
    test_data = next(iter(test_load))[0].numpy()
    train_data = train_data[idx_train]
    test_data = test_data[idx_test]
    train_label = train_label[idx_train]
    test_label = test_label[idx_test]
    
    
    model = Net(num_classes=args.num_classes, classifier_type=args.classifier_type).to(device)
    c = 0
    for m in args.model_path:
        attribution_method = attribution_methods.AttributionMethods(args.ROAR, args.occlude, 28, args.attribution_method,
            replacement=args.replacement_type, dataset_min=-0.1307/0.3081, custom_value=args.replacement_value) 
    
        model.load_state_dict(torch.load(m, map_location=device))
        #test_raw_data(args, model, device, test_data, test_label)
        c += test_attributed_data(args, model, device, test_data, test_label, attribution_method)
        #plot_images(args, model, device, test_data, test_label, attribution_method, num=10, indices = np.array([1,2,3,4,5,6,7,8,9,10]))
    print(c/5)       
if __name__=="__main__":
    
    main()
