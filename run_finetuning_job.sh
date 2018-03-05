#!/bin/bash

# SLURM_JOBID
WORKDIR=$1
DATADIR=$2
cd $WORKDIR
source setup_container.sh
rsync $DATADIR/*.root /tmp/

jobdir=`printf slurm_finetune_%d ${SLURMJOBID}`
mkdir $jobdir
cp finetune_wlarcv1_cosmics.py ${jobdir}/
cp caffe_uresnet.py ${jobdir}/

cd $jobdir
mkdir runs

python finetune_wlarcv1_cosmics.py
