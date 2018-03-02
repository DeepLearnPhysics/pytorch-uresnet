
workdir=$PWD
source /usr/local/root/release/bin/thisroot.sh
cd /usr/local/larcv
source configure.sh
export PATH=/usr/local/nvidia:${PATH}
export LD_LIBRARY_PATH=/usr/local/nvidia:${LD_LIBRARY_PATH}
cd $workdir