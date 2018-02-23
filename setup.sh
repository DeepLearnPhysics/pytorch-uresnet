#!/bin/bash

home=$PWD
source ~/setup_root6.sh
source ~/setup_cuda.sh

#cd ../larcv2
#cd ../larcv1
export DLLEE_UNIFIED_BASEDIR=~/working/larbys/dllee_unified
cd ~/working/larbys/dllee_unified/
source configure.sh

cd $home

cd ../larcvdataset
source setenv.sh

cd $home

export CUDA_VISIBLE_DEVICES=0,1
