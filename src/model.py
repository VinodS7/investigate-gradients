#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network architecture definition for Singing Voice Detection experiment.

Author: Jan Schlüter
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import audio

###############################################################################
#-------------Pytorch implementation of singing voice detection---------------#
###############################################################################
def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            # if param.dim() > 1:
            #     print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            # else:
            #     print(name, ':', num_param)
            total_param += num_param
    # print('Total Parameters: {}'.format(total_param))
    return total_param

class STFTLayer(nn.Module):
    def __init__(self, sample_rate, frame_len, fps, device):
        
        super(STFTLayer, self).__init__()
        self.device = device
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.fps = fps

    def forward(self,x):
        sp = torch.stft(x, n_fft=self.frame_len,
            hop_length=self.sample_rate // self.fps, win_length=self.frame_len,
            window=torch.hann_window(self.frame_len, periodic=False, requires_grad=True).to(self.device),
            center=False).transpose(1,2)
        sp = torch.sqrt(torch.square(sp[:,:,:,0]) + torch.square(sp[:,:,:,1]) + 1e-8)
        return sp



class LogMelBankLayer(nn.Module):
    def __init__(self, sample_rate, frame_len, mel_bands, mel_min, mel_max,
                bin_mel_max, device):
        
        super(LogMelBankLayer, self).__init__()
        self.device = device
        self.bin_mel_max = bin_mel_max
        self.filterbank = audio.create_mel_filterbank(sample_rate, frame_len,
                    mel_bands, mel_min, mel_max)
        self.filterbank = torch.Tensor(self.filterbank[:bin_mel_max].astype(np.float32)).to(device)

    def forward(self,x):
         
        filt = self.filterbank.repeat([x.size(0),1,1])
        x = torch.bmm(x[:,:,:self.bin_mel_max],filt)
        x = torch.max(x,torch.Tensor([1e-7]).to(self.device))
        x = torch.log(x)
        #x = torch.log(torch.max(torch.dot(x[:,:self.bin_mel_max],self.filterbank), 1e-7))
        x = x.unsqueeze(1)
        #x = x.unsqueeze(0)

        #x = (x-self.mean)/self.std
        return x

class CNNModel(nn.Module):
    def __init__(self,model_type='CNN', input_type='baseline',is_zeromean=False, sample_rate=None, frame_len=None, fps=None, mel_bands=None, mel_min=None, mel_max=None, bin_mel_max=None, meanstd_file=None, device=None):
        super(CNNModel, self).__init__()
        self.model_type = model_type
        self.input_type = input_type
        self.meanstd_file = meanstd_file
        self.is_zeromean = is_zeromean
        if(input_type=='audio'):
            self.stft = STFTLayer(sample_rate, frame_len, fps, device)
            self.logmel = LogMelBankLayer(sample_rate, frame_len, mel_bands, mel_min, mel_max, bin_mel_max, device)
        elif(input_type=='stft'):
            self.logmel = LogMelBankLayer(sample_rate, frame_len, mel_bands, mel_min, mel_max, bin_mel_max, device)
        with np.load(meanstd_file) as f:
            self.mean = torch.Tensor(f['mean']).to(device)
            self.istd = (1.0/torch.Tensor(f['std']).to(device))


        if(self.model_type=='CRNN'):
            self.conv = nn.Sequential(
                    nn.Conv2d(1,64,3),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(64,32,3),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool2d(3),
                    nn.Conv2d(32,128,3),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(128,64,3),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool2d(3))
        
            #self.recur = nn.Sequential(
            #        nn.Flatten(2,-1),
            #        nn.GRU(64*7, 32, batch_first=True))
            #self.fc = nn.Sequential(
            #        nn.Flatten(),
            #        nn.Dropout(),
            #        nn.Linear(32*11,64),
            #        nn.LeakyReLU(inplace=True),
            #        nn.Dropout(),
            #        nn.Linear(64,1),
            #        nn.Sigmoid())

            # head
            #self.recur = nn.Sequential(
            #        nn.Flatten(2,-1),
            #        nn.GRU(64*7, 16, batch_first=True))
            #self.fc = nn.Sequential(
            #        nn.Flatten(),
            #        nn.Linear(16,1),
            #        nn.Sigmoid())

            # bi-directional GRU 
            self.recur = nn.Sequential(
                    nn.Flatten(2,-1),
                    nn.GRU(64*7, 16, batch_first=True,
                        bidirectional=True))
            self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32,1),
                    nn.Sigmoid())


        elif(model_type=='CNN'):
            self.conv = nn.Sequential(
                    nn.Conv2d(1,64,3),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(64,32,3),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool2d(3),
                    nn.Conv2d(32,128,3),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(128,64,3),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool2d(3),
                    nn.Flatten())
        
            self.fc = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(64*11*7,256),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(256,64),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(64,1),
                    nn.Sigmoid())

        self.param_count = count_parameters(self)
        print(self.param_count)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self,x):
        #print(self.training)
        if(self.input_type=='audio'):
            x = self.stft(x)
            x = self.logmel(x)
            x = (x-self.mean) * self.istd
            
        if(self.input_type=='stft'):
            x = self.logmel(x)
            x = (x-self.mean) * self.istd
            
        elif(self.input_type=='mel_spects'):
            x = (x-self.mean) * self.istd
            #x = x.unsqueeze(1)

        if(self.is_zeromean):
            self.conv[0].weight.data = self.conv[0].weight.data - torch.mean(self.conv[0].weight.data,(2,3),keepdim=True)
        if(self.model_type=='CNN'):
            x = self.conv(x)
            return self.fc(x)
        elif(self.model_type=='CRNN'):
            x = self.conv(x)
            x = torch.transpose(x,1,2)
            x, _ = self.recur(x)
            
            # head of GRU output
            #x = x[:,-1,:]

            # bi-directional GRU mid
            x = x[:,5,:]
            
            x = self.fc(x)
            return x

class CNNTwoClass(nn.Module):
    def __init__(self):
        super(CNNTwoClass, self).__init__()

        self.conv = nn.Sequential(
                    nn.Conv2d(1,64,3),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(64,32,3),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool2d(3),
                    nn.Conv2d(32,128,3),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(128,64,3),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool2d(3),
                    nn.Flatten())
        
        self.fc = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(64*11*7,256),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(256,64),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(64,2))
        

        self.param_count = count_parameters(self)
        print(self.param_count)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                                 
    
    def forward(self,x):
        x = self.conv(x)
        return self.fc(x)



class Net(nn.Module):
    def __init__(self, num_classes=10, classifier_type='softmax'):
        super(Net, self).__init__()
        self.classifier_type = classifier_type
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        if(classifier_type=='softmax'):
            self.fc2 = nn.Linear(128, num_classes)
        else:
            self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        if(self.classifier_type=='softmax'):
            output = F.log_softmax(x, dim=1)
        elif(self.classifier_type=='sigmoid'):
            output = F.sigmoid(x)
        return output
 









































