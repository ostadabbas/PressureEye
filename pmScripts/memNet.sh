#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --job-name=vis2PM
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=100Gb
#SBATCH --output=bch_outs/memNet_RGB_clip01.%j.out
#SBATCH --error=bch_outs/memNet_RGB_clip01.%j.err
source activate pch1.5
python train.py --dataroot /scratch/liu.shu/datasets/SLP/danaLab \
--model memNet --mod_src ${1} --mod_tar PMarray --batch_size 2 \

