#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=9:30:00
#SBATCH --job-name=vis2PM
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=100Gb
#SBATCH --output=bch_outs/pix2pix_RGB_clip11.%j.out
#SBATCH --error=bch_outs/pix2pix_RGB_clip11.%j.err
source activate pch1.5
python train.py --dataroot /scratch/liu.shu/datasets/SLP/danaLab \
--model pix2pix --mod_src ${1} --mod_tar PMarray --pmDsProc clip11 --batch_size 70 --lambda_D 1 \


# the result suppose to be pix2pix