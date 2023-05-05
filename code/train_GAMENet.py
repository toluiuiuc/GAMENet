import torch
import argparse
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict

from models import GAMENet
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params

from gamenet_main import main

torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'GAMENet'
resume_name = ''

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')
parser.add_argument('--ddi', action='store_true', default=False, help="using ddi")
parser.add_argument('--remove_dm', type=str, default=None, help='remove DM method')
parser.add_argument('--cpu', action='store_true', default=False, help="CPU mode")
parser.add_argument('--graph_type', type=str, default='GCN', help="Graph Type")

args = parser.parse_args()
model_name = args.model_name
resume_name = args.resume_path

if __name__ == '__main__':
    main(cpu=args.cpu, isTest=args.eval, withDDI=args.ddi, graph_type=args.graph_type, remove_dm=args.remove_dm, model_name=model_name, resume_name=resume_name)
