#!/bin/bash
# get result from AR pc via scp
#echo scp -r ${1} liu.shu@xfer.discovery.neu.edu:/scratch/liu.shu/codesPool/PEye/
#scp -r ${1} liu.shu@xfer.discovery.neu.edu:/scratch/liu.shu/codesPool/PEye/
echo scp -rq jun@129.10.132.45:~/codesPool/PEye/results/${1} results
scp -r jun@129.10.132.45:~/codesPool/PEye/results/${1} results