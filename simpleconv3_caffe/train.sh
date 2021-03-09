#########################################################################
# Author: longpeng2008to2012@gmail.com
#########################################################################
SOLVER=./solver.prototxt
/home/rick/Caffe_Long/build/tools/caffe train -solver $SOLVER -gpu 0 2>&1 | tee log.txt 
