#!/bin/env python

## IMPORT

# python,numpy
import os,sys,commands
import shutil
import time
import traceback
import numpy as np

# ROOT, larcv
import ROOT
from larcv import larcv

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

# tensorboardX
from tensorboardX import SummaryWriter

# Our model definition
from uresnet import UResNet

GPUMODE=True


# SegData: class to hold batch data
# we expect LArCV1Dataset to fill this object
class SegData:
    def __init__(self):
        self.dim = None
        self.images = None # adc image
        self.labels = None # labels
        self.weights = None # weights
        return

    def shape(self):
        if self.dim is None:
            raise ValueError("SegData instance hasn't been filled yet")
        return self.dim
    
    

# Data interface
class LArCV1Dataset:
    def __init__(self, name, cfgfile ):
        # inputs
        # cfgfile: path to configuration. see test.py.ipynb for example of configuration
        self.name = name
        self.cfgfile = cfgfile
        return
      
    def init(self):
        # create instance of data file interface
        self.io = larcv.ThreadDatumFiller(self.name)
        self.io.configure(self.cfgfile)
        self.nentries = self.io.get_n_entries()
        self.io.set_next_index(0)
        print "[LArCV1Data] able to create ThreadDatumFiller"
        return
        
    def getbatch(self, batchsize):
        self.io.batch_process(batchsize)
        time.sleep(0.1)
        itry = 0
        while self.io.thread_running() and itry<100:
            time.sleep(0.01)
            itry += 1
        if itry>=100:
            raise RuntimeError("Batch Loader timed out")
        
        # fill SegData object
        data = SegData()
        dimv = self.io.dim() # c++ std vector through ROOT bindings
        self.dim     = (dimv[0], dimv[1], dimv[2], dimv[3] )
        self.dim3    = (dimv[0], dimv[2], dimv[3] )

        # numpy arrays
        data.np_images  = np.zeros( self.dim,  dtype=np.float32 )
        data.np_labels  = np.zeros( self.dim3, dtype=np.int )
        data.np_weights = np.zeros( self.dim3, dtype=np.float32 )
        data.np_images[:]  = larcv.as_ndarray(self.io.data()).reshape(    self.dim  )[:]
        data.np_labels[:]  = larcv.as_ndarray(self.io.labels()).reshape(  self.dim3 )[:]
        data.np_weights[:] = larcv.as_ndarray(self.io.weights()).reshape( self.dim3 )[:]
        data.np_labels[:] += -1
        
        # pytorch tensors
        data.images = torch.from_numpy(data.np_images)
        data.labels = torch.from_numpy(data.np_labels)
        data.weight = torch.from_numpy(data.np_weights)
        #if GPUMODE:
        #    data.images.cuda()
        #    data.labels.cuda(async=False)
        #    data.weight.cuda(async=False)


        # debug values
        #print "max label: ",np.max(data.labels)
        #print "min label: ",np.min(data.labels)
        
        return data

# Data augmentation/manipulation functions
def padandcrop(npimg2d):
    imgpad  = np.zeros( (264,264), dtype=np.float32 )
    imgpad[4:256+4,4:256+4] = npimg2d[:,:]
    randx = np.random.randint(0,8)
    randy = np.random.randint(0,8)
    return imgpad[randx:randx+256,randy:randy+256]

def padandcropandflip(npimg2d):
    imgpad  = np.zeros( (264,264), dtype=np.float32 )
    imgpad[4:256+4,4:256+4] = npimg2d[:,:]
    if np.random.rand()>0.5:
        imgpad = np.flip( imgpad, 0 )
    if np.random.rand()>0.5:
        imgpad = np.flip( imgpad, 1 )
    randx = np.random.randint(0,8)
    randy = np.random.randint(0,8)
    return imgpad[randx:randx+256,randy:randy+256]

# Loss Function
# We define a pixel wise L2 loss

# taken from torch.nn.modules.loss
def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"
        
class PixelWiseNLLLoss(nn.modules.loss._WeightedLoss):
    def __init__(self,weight=None, size_average=True, ignore_index=-100 ):
        super(PixelWiseNLLLoss,self).__init__(weight,size_average)
        self.ignore_index = ignore_index
        self.reduce = False
        #self.mean = torch.mean.cuda()

    def forward(self,predict,target,pixelweights):
        """
        predict: (b,c,h,w) tensor with output from logsoftmax
        target:  (b,h,w) tensor with correct class
        pixelweights: (b,h,w) tensor with weights for each pixel
        """
        _assert_no_grad(target)
        _assert_no_grad(pixelweights)
        # reduce for below is false, so returns (b,h,w)
        pixelloss = F.nll_loss(predict,target, self.weight, self.size_average, self.ignore_index, self.reduce)
        pixelloss *= pixelweights
        return torch.mean(pixelloss)
        

torch.cuda.device( 1 )

# global variables
best_prec1 = 0.0  # best accuracy, use to decide when to save network weights
writer = SummaryWriter()

def main():

    global best_prec1
    global writer

    # create model, mark it to run on the GPU
    if GPUMODE:
        model = UResNet(inplanes=16,input_channels=1,num_classes=3)
        model.cuda()
    else:
        model = UResNet(inplanes=16,input_channels=1,num_classes=3)

    # uncomment to dump model
    #print "Loaded model: ",model
    # check where model pars are
    #for p in model.parameters():
    #    print p.is_cuda


    # define loss function (criterion) and optimizer
    if GPUMODE:
        criterion = PixelWiseNLLLoss().cuda()
    else:
        criterion = PixelWiseNLLLoss()

    # training parameters
    lr = 1.0e-3
    momentum = 0.9
    weight_decay = 1.0e-3

    # training length
    batchsize_train = 1
    batchsize_valid = 1
    start_epoch = 0
    epochs      = 1
    start_iter  = 0
    num_iters   = 10000
    #num_iters    = None # if None
    iter_per_epoch = None # determined later

    nbatches_per_itertrain = 4
    itersize_train         = batchsize_train*nbatches_per_itertrain
    trainbatches_per_print = 1
    
    nbatches_per_itervalid = 4
    itersize_valid         = batchsize_valid*nbatches_per_itervalid
    validbatches_per_print = 1

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    cudnn.benchmark = True

    # LOAD THE DATASET

    # define configurations
    traincfg = """ThreadDatumFillerTrain: {

  Verbosity:    2
  EnableFilter: false
  RandomAccess: true
  UseThread:    false
  InputFiles:   ["/mnt/raid0/taritree/ssnet_training_data/train00.root","/mnt/raid0/taritree/ssnet_training_data/train01.root"]  
  ProcessType:  ["SegFiller"]
  ProcessName:  ["SegFiller"]

  IOManager: {
    Verbosity: 2
    IOMode: 0
    ReadOnlyTypes: [0,0,0]
    ReadOnlyNames: ["wire","segment","ts_keyspweight"]
  }
    
  ProcessList: {
    SegFiller: {
      # DatumFillerBase configuration
      Verbosity: 2
      ImageProducer:     "wire"
      LabelProducer:     "segment"
      WeightProducer:    "ts_keyspweight"
      # SegFiller configuration
      Channels: [2]
      SegChannel: 2
      EnableMirror: false
      EnableCrop: false
      ClassTypeList: [0,1,2]
      ClassTypeDef: [0,0,0,2,2,2,1,1,1,1]
    }
  }
}
"""
    validcfg = """ThreadDatumFillerValid: {

  Verbosity:    2
  EnableFilter: false
  RandomAccess: true
  UseThread:    false
  InputFiles:   ["/mnt/raid0/taritree/ssnet_training_data/train02.root"]  
  ProcessType:  ["SegFiller"]
  ProcessName:  ["SegFiller"]

  IOManager: {
    Verbosity: 2
    IOMode: 0
    ReadOnlyTypes: [0,0,0]
    ReadOnlyNames: ["wire","segment","ts_keyspweight"]
  }
    
  ProcessList: {
    SegFiller: {
      # DatumFillerBase configuration
      Verbosity: 2
      ImageProducer:     "wire"
      LabelProducer:     "segment"
      WeightProducer:    "ts_keyspweight"
      # SegFiller configuration
      Channels: [2]
      SegChannel: 2
      EnableMirror: false
      EnableCrop: false
      ClassTypeList: [0,1,2]
      ClassTypeDef: [0,0,0,2,2,2,1,1,1,1]
    }
  }
}
"""
    with open("segfiller_train.cfg",'w') as ftrain:
        print >> ftrain,traincfg
    with open("segfiller_valid.cfg",'w') as fvalid:
        print >> fvalid,validcfg
    
    iotrain = LArCV1Dataset("ThreadDatumFillerTrain","segfiller_train.cfg" )
    iovalid = LArCV1Dataset("ThreadDatumFillerValid","segfiller_valid.cfg" )
    iotrain.init()
    iovalid.init()
    iotrain.getbatch(batchsize_train)

    NENTRIES = iotrain.io.get_n_entries()
    print "Number of entries in training set: ",NENTRIES

    if NENTRIES>0:
        iter_per_epoch = NENTRIES/(itersize_train)
        if num_iters is None:
            # we set it by the number of request epochs
            num_iters = (epochs-start_epoch)*NENTRIES
        else:
            epochs = num_iters/NENTRIES
    else:
        iter_per_epoch = 1

    print "Number of epochs: ",epochs
    print "Iter per epoch: ",iter_per_epoch

    with torch.autograd.profiler.profile(enabled=False) as prof:

        # Resume training option
        if False:
            checkpoint = torch.load( "checkpoint.pth.p01.tar" )
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint['optimizer'])
        

        for ii in range(start_iter, num_iters):

            adjust_learning_rate(optimizer, ii, lr)
            print "Iter:%d Epoch:%d.%d "%(ii,ii/iter_per_epoch,ii%iter_per_epoch),
            for param_group in optimizer.param_groups:
                print "lr=%.3e"%(param_group['lr']),
                print

            # train for one epoch
            try:
                train_ave_loss, train_ave_acc = train(iotrain, batchsize_train, model,
                                                      criterion, optimizer,
                                                      nbatches_per_itertrain, ii, trainbatches_per_print)
            except Exception,e:
                print "Error in training routine!"            
                print e.message
                print e.__class__.__name__
                traceback.print_exc(e)
                break
            print "Iter:%d Epoch:%d.%d train aveloss=%.3f aveacc=%.3f"%(ii,ii/iter_per_epoch,ii%iter_per_epoch,train_ave_loss,train_ave_acc)

            # evaluate on validation set
            try:
                prec1 = validate(iovalid, batchsize_valid, model, criterion, nbatches_per_itervalid, validbatches_per_print, ii)
            except Exception,e:
                print "Error in validation routine!"            
                print e.message
                print e.__class__.__name__
                traceback.print_exc(e)
                break

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'iter':ii,
                'epoch': ii/iter_per_epoch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, -1)
            if ii>0 and ii%iter_per_epoch==0:
                save_checkpoint({
                    'iter':ii,
                    'epoch': ii/iter_per_epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, False, epoch)

    print "FIN"
    print "PROFILER"
    print prof
    writer.close()


def train(train_loader, batchsize, model, criterion, optimizer, nbatches, epoch, print_freq):

    global writer
    
    batch_time = AverageMeter() # total for batch
    data_time = AverageMeter()
    format_time = AverageMeter()
    train_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    model.cuda()

    for i in range(0,nbatches):
        #print "epoch ",epoch," batch ",i," of ",nbatches
        batchstart = time.time()
    
        end = time.time()        
        data = train_loader.getbatch(batchsize)
        # measure data loading time
        data_time.update(time.time() - end)

        end = time.time()

        # measure data formatting time
        format_time.update(time.time() - end)
        
        # convert to pytorch Variable (with automatic gradient calc.)
        if GPUMODE:
            images_var = torch.autograd.Variable(data.images.cuda())
            labels_var = torch.autograd.Variable(data.labels.cuda(),requires_grad=False)
            weight_var = torch.autograd.Variable(data.weight.cuda(),requires_grad=False)
        else:
            images_var = torch.autograd.Variable(data.images)
            labels_var = torch.autograd.Variable(data.labels,requires_grad=False)
            weight_var = torch.autograd.Variable(data.weight,requires_grad=False)
            
        # compute output
        end = time.time()
        output = model(images_var)
        loss = criterion(output, labels_var, weight_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, labels_var.data)
        losses.update(loss.data[0], data.images.size(0))
        top1.update(prec1[-1], data.images.size(0))
        writer.add_scalar('data/train_loss', loss.data[0], epoch )        
        writer.add_scalars('data/train_accuracy', {'background': prec1[0],
                                                   'track': prec1[1],
                                                   'shower': prec1[2],
                                                   'total':prec1[3]}, epoch )

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_time.update(time.time()-end)

        # measure elapsed time
        batch_time.update(time.time() - batchstart)


        if i % print_freq == 0:
            status = (epoch,i,nbatches,
                      batch_time.val,batch_time.avg,
                      data_time.val,data_time.avg,
                      format_time.val,format_time.avg,
                      train_time.val,train_time.avg,
                      losses.val,losses.avg,
                      top1.val,top1.avg)
            print "Epoch: [%d][%d/%d]\tTime %.3f (%.3f)\tData %.3f (%.3f)\tFormat %.3f (%.3f)\tTrain %.3f (%.3f)\tLoss %.3f (%.3f)\tPrec@1 %.3f (%.3f)"%status
            #print('Epoch: [{0}][{1}/{2}]\t'
            #      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #      'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
            #      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #          epoch, i, len(train_loader), batch_time=batch_time,
            #          data_time=data_time, losses=losses, top1=top1 ))
    return losses.avg,top1.avg


def validate(val_loader, batchsize, model, criterion, nbatches, print_freq, iiter):

    global writer
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i in range(0,nbatches):
        data = val_loader.getbatch(batchsize)

        # convert to pytorch Variable (with automatic gradient calc.)
        if GPUMODE:
            images_var = torch.autograd.Variable(data.images.cuda())
            labels_var = torch.autograd.Variable(data.labels.cuda(),requires_grad=False)
            weight_var = torch.autograd.Variable(data.weight.cuda(),requires_grad=False)
        else:
            images_var = torch.autograd.Variable(data.images)
            labels_var = torch.autograd.Variable(data.labels,requires_grad=False)
            weight_var = torch.autograd.Variable(data.weight,requires_grad=False)

        # compute output
        output = model(images_var)
        loss = criterion(output, labels_var, weight_var)
        #loss = criterion(output, labels_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, labels_var.data)
        losses.update(loss.data[0], data.images.size(0))
        top1.update(prec1[-1], data.images.size(0))
        writer.add_scalar('data/valid_loss', loss.data[0], iiter )
        writer.add_scalars('data/valid_accuracy', {'background': prec1[0],
                                                   'track': prec1[1],
                                                   'shower': prec1[2],
                                                   'total':prec1[3]}, iiter )

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            status = (i,nbatches,batch_time.val,batch_time.avg,losses.val,losses.avg,top1.val,top1.avg)
            print "Test: [%d/%d]\tTime %.3f (%.3f)\tLoss %.3f (%.3f)\tPrec@1 %.3f (%.3f)"%status
            #print('Test: [{0}/{1}]\t'
            #      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #        i, len(val_loader), batch_time=batch_time, loss=losses,
            #        top1=top1))

    #print(' * Prec@1 {top1.avg:.3f}'
    #      .format(top1=top1))
    print "Test:Result* Prec@1 %.3f\tLoss %.3f"%(top1.avg,losses.avg)

    return float(top1.avg)


def save_checkpoint(state, is_best, p, filename='checkpoint.pth.tar'):
    if p>0:
        filename = "checkpoint.%dth.tar"%(p)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.tar')


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


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = lr * (0.5 ** (epoch // 300))
    lr = lr
    #lr = lr*0.992
    #print "adjust learning rate to ",lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """Computes the accuracy. we want the aggregate accuracy along with accuracies for the different labels. easiest to just use numpy..."""
    maxk = 1
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, False)

    correct = pred.eq(target)

    np_correct = correct.cpu().numpy()
    np_target  = target.cpu().numpy()
    print np_correct.shape
    print np_target.shape

    unique, count = np.unique( np_target, return_counts=True )
    denom = dict(zip(unique,count))
    print unique,count
    
    res = []
    for c in range(output.size(1)):
        if c not in unique:
            res.append(0)
            continue
        cc = np.sum( np_correct[ np_target.reshape(np_correct.shape)==c ] )
        res.append( float(cc)/float(denom[c])*100.0 )

    # totals
    res.append( 100.0*np.sum(np_correct)/np_correct.size )
        
    return res

def dump_lr_schedule( startlr, numepochs ):
    for epoch in range(0,numepochs):
        lr = startlr*(0.5**(epoch//300))
        if epoch%10==0:
            print "Epoch [%d] lr=%.3e"%(epoch,lr)
    print "Epoch [%d] lr=%.3e"%(epoch,lr)
    return

if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
