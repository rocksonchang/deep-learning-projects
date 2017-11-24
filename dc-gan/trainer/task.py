#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper function that passes command line inputs to the actual trainer for GCloud ML.
Supports both python2 and python3.

Must config model_config_batch4_c.yaml first before running model.
All parameters and network architecture are configured in that file!


----------
Created on Wed Nov 8
@author: Rockson
"""
from __future__ import absolute_import, division, print_function
import argparse

from trainer import dcgan


if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    # Argument required by GC
    parser.add_argument('--job-dir', help='GCS location to write checkpoints and export models', 
                        required=True)    

    parser.add_argument('--n_epochs', help='Number of epochs', 
                        default=50, type=int)

    parser.add_argument('--dataset', help='Dataset: mnist or mnistf', 
                        default='mnist', type=str)
    
    parser.add_argument('--BATCH_SIZE', help='Batch size', 
                        default=128, type=int)

    parser.add_argument('--BUCKET_NAME', help='GCS bucket name', 
                        default='rc_bucket', type=str)

    args = parser.parse_args()
    arguments = args.__dict__

    dcgan.train(**arguments)   

