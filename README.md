# pytorch-uresnet

PyTorch Implementation of U-ResNet used for track/shower pixel-labeling

## Dependencies

* `ROOT`: data analysis framework. Defines file format, provides python bindings for our code
* `LArCV`: either version 1 and 2
* `pytorch`: network implementation
* `tensorboardX`: interface to log data that can be plotted with Tensorboard
* `tensorflow-tensorboard`: (from tensorflow) for plotting loss/accuracy
* `jupyter notebook`: for testing code and inspecting images

### Known working configuration

  * ubuntu 16.10, ROOT 6, python 2.7.12, tensorflow-tensorboard (from tensorflow 1.4.1), cuda 8.0

## Files

* `uresnet.py`: module that defines the network
* `plotlarcv1.ipynb`: notebook used to test file IO commands. For LArCV classic files.
* `test.py.ipynb`: notebook used to test file IO commands. For LArCV2 files.
* `train_wlarcv1.py`: script for training. 
