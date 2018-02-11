import os,sys
import ROOT as rt
from larcv import larcv
from uresnet import UResNet
from larcvdataset import LArCVDataset

#net = UResNet( num_classes=3, input_channels=1, inplanes=16 )

# we load in a test image
#iotest = LArCVDataset("test_dataloader.cfg", "ThreadProcessorTest")
iotest = LArCVDataset("test_threadfiller.cfg", "ThreadProcessorTest")
iotest.start(1)

data = iotest[0]
print data
#print net
iotest.stop()
