#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

from __future__ import print_function

import sys
import os
import io

import numpy as np


class AttributionMethods():
    def __init__(self, ROAR, occlude, mel_bins, attribution, replacement='mean', dataset_min=0, dataset_mean=0, custom_value=0):
        self.ROAR = ROAR
        self.occlude = occlude
        self.mel_bins = mel_bins
        self.attribution = attribution
        self.replacement = replacement
        
        self.dataset_min = dataset_min
        self.dataset_mean = dataset_mean
        self.custom_value = custom_value
        
        self.ctrb_sgns = np.zeros((3,3))
        return
    def contribution_signs(self, input_data, grad=None):
        contribution_signs = np.zeros((3,3))
        _, v = self.apply_attribution(input_data.copy(), grad.copy())
        temp_data = input_data[:, v//self.mel_bins, v%self.mel_bins].flatten()

 
        if self.rank_matrix is None:
            print('Not applicable')
            return None

        temp_grad = self.rank_matrix[:, v//self.mel_bins, v%self.mel_bins].flatten()

        for i in range(len(temp_data)):
            if(temp_data[i]>0):
                ind1=0
            elif(temp_data[i]<0):
                ind1=1
            elif(temp_data[i]==0):
                ind1=2
    
            if(temp_grad[i]>0):
                ind2=0
            elif(temp_grad[i]<0):
                ind2=1
            elif(temp_grad[i]==0):
                ind2=2
                
            contribution_signs[ind1][ind2]+=1
        self.ctrb_sgns+= contribution_signs #/np.sum(contribution_signs)
        return contribution_signs
                

    def apply_attribution(self,input_data, grad=None):
        if(self.attribution=='random'):
            v = self.random(input_data)
        elif(self.attribution=='abs_grad'):
            v = self.abs_grad(grad)
        elif(self.attribution=='grad_orig'):
            v = self.grad_orig(grad)
        elif(self.attribution=='grad_inp'):
            v = self.grad_inp(input_data, grad)
        elif(self.attribution=='abs_grad_inp'):
            v = self.abs_grad_inp(input_data, grad)
        elif(self.attribution=='grad_abs_inp'):
            v = self.grad_abs_inp(input_data, grad)
        elif(self.attribution=='abs_grad_abs_inp'):
            v = self.abs_grad_abs_inp(input_data, grad)
        elif(self.attribution=='inp'):
            v = self.inp(input_data)
        elif(self.attribution=='abs_inp'):
            v = self.abs_inp(input_data)


        if(self.replacement=='mean'):
            input_data[:, v//self.mel_bins, v%self.mel_bins] = self.dataset_mean
        elif(self.replacement=='dataset_min'):
            input_data[:,v//self.mel_bins, v%self.mel_bins] = self.dataset_min
        elif(self.replacement=='custom'):
            input_data[:,v//self.mel_bins, v%self.mel_bins] = self.custom_value
        elif(self.replacement=='image_min'):
            val = np.min(input_data)
            input_data[:, v//self.mel_bins, v%self.mel_bins] = 3*val
        elif(self.replacement=='image_max'):
            val = np.max(input_data)
            input_data[:, v//self.mel_bins, v%self.mel_bins] = val
        
        return input_data, v



    def random(self, input_data):
        v = np.random.choice(input_data.shape[1]*input_data.shape[2], self.occlude, replace=False)
        return v

    def abs_grad(self, grad):
        self.rank_matrix = np.abs(grad)
        
        if(self.ROAR==1):
            v = np.argsort(self.rank_matrix, axis=None)[-self.occlude:]
        else:
            v = np.argsort(self.rank_matrix, axis=None)[:self.occlude]
        return v

    def grad_orig(self, grad):
        self.rank_matrix = grad
        if(self.ROAR==1):
            v = np.argsort(self.rank_matrix, axis=None)[-self.occlude:]
        else:
            v = np.argsort(self.rank_matrix, axis=None)[:self.occlude]
        return v

    
    def grad_inp(self, input_data, grad):
        self.rank_matrix = np.squeeze(grad*input_data)
        if(self.ROAR==1):
            v = np.argsort(self.rank_matrix, axis=None)[-self.occlude:]
        else:
            v = np.argsort(self.rank_matrix, axis=None)[:self.occlude]
        return v

    def abs_grad_inp(self, input_data, grad):
        self.rank_matrix = np.squeeze(np.abs(grad)*input_data)
        if(self.ROAR==1):
            v = np.argsort(self.rank_matrix, axis=None)[-self.occlude:]
        else:
            v = np.argsort(self.rank_matrix, axis=None)[:self.occlude]
        return v

    def grad_abs_inp(self, input_data, grad):
        self.rank_matrix = np.squeeze(grad*np.abs(input_data))
        if(self.ROAR==1):
            v = np.argsort(self.rank_matrix, axis=None)[-self.occlude:]
        else:
            v = np.argsort(self.rank_matrix, axis=None)[:self.occlude]
        return v

    def abs_grad_abs_inp(self, input_data, grad):
        self.rank_matrix = np.squeeze(np.abs(grad)*np.abs(input_data))
        if(self.ROAR==1):
            v = np.argsort(self.rank_matrix, axis=None)[-self.occlude:]
        else:
            v = np.argsort(self.rank_matrix, axis=None)[:self.occlude]
        return v

    def inp(self, input_data):
        self.rank_matrix = np.squeeze(input_data)
        if(self.ROAR==1):
            v = np.argsort(self.rank_matrix, axis=None)[-self.occlude:]
        else:
            v = np.argsort(self.rank_matrix, axis=None)[:self.occlude]
        return v

    def abs_inp(self, input_data):
        self.rank_matrix = np.squeeze(np.abs(input_data))
        if(self.ROAR==1):
            v = np.argsort(self.rank_matrix, axis=None)[-self.occlude:]
        else:
            v = np.argsort(self.rank_matrix, axis=None)[:self.occlude]
        return v
