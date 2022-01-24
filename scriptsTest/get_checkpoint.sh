#!/bin/bash
# test echo first
#echo scp -r ${1} liu.shu@xfer.discovery.neu.edu:/scratch/liu.shu/codesPool/PEye/
#scp -r ${1} liu.shu@xfer.discovery.neu.edu:/scratch/liu.shu/codesPool/PEye/
echo scp -rq jun@129.10.132.45:~/codesPool/PEye/checkpoints/${1} checkpoints
scp -r jun@129.10.132.45:~/codesPool/PEye/checkpoints/${1} checkpoints