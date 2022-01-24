#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=1:00:00
#SBATCH --job-name=vis2PM
#SBATCH --partition=general
#SBATCH --mem=100Gb
#SBATCH --output=bch_outs/sbatch_test.%j.out
#SBATCH --error=bch_outs/sbatch_test.%j.err

#source activate py36
#python train.py --dataroot /scratch/liu.shu/datasets/datasetPM/danaLab \
#--name ts1 --dataset_mode pm --display_freq 1 --n_train 1 --num_test_in_train 1 --num_test 1 --niter 1 --niter_decay 0 --if_saveDiff --if_saveImg --display_id 1 \
#--model vis2PM --mod_src RGB --mod_tar PM --n_phy 1 --batch_size 30 \
#--type_whtL n --lambda_sum 1e-8 --lambda_ssim 10 --if_D D \
#--suffix {pmDsProc}_{if_align}-align_{n_phy}_{n_stg}stg_{if_actiFn}actiFn_{lambda_L}{type_L}_{type_whtL}-whtL{whtScal}_{lambda_sum}sum_{lambda_ssim}ssim_{if_D}{n_layers_D}_epoch{niter}_{niter_decay}_bch{batch_size}
# test if can pass parameters in with sbatch
#echo ${1} # can pass in

#  check the condiiton to call name of functions
if [ ${1} = train ]; then
  if_saveImg=n
elif [ ${1} = testPM ]; then
  if_saveImg=y  # can assign multiple value
else
  echo no such options for PE choose train\|testPM instead
  exit 1
fi
echo python ${1}.py --if_saveImg $if_saveImg