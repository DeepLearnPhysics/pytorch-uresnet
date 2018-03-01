import os,sys,time
import ROOT
from ROOT import std
from larcv import larcv
import numpy as np

import torch
import torch.nn

from caffe_uresnet import UResNet

GPUMODE=True
GPUID=1

net = UResNet(inplanes=16,input_channels=1,num_classes=3)
if GPUMODE:
    net.cuda(GPUID)

weightfile = "checkpoint.20000th.tar"
checkpoint = torch.load(weightfile)
net.load_state_dict(checkpoint["state_dict"])

print "[ENTER] to end"
raw_input()
