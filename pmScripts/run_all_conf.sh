#!/bin/bash
# ran with configuration
# task: only sum version, auto_sum already
sbatch --exclude=d1005 pmScripts/clip01.sh ${1}
sbatch --exclude=d1005 pmScripts/clip01_autoW.sh ${1}
sbatch --exclude=d1005 pmScripts/clip01_pwrs0.sh ${1}
sbatch --exclude=d1005 pmScripts/clip01_autoW_sum.sh ${1}
sbatch --exclude=d1005 pmScripts/clip01_autoW_sum_ssim.sh ${1}
sbatch --exclude=d1005 pmScripts/clip01_autoW_sum_ssim_D.sh ${1}
sbatch --exclude=d1005 pmScripts/clip01_D.sh ${1}
sbatch --exclude=d1005 pmScripts/clip01_nPhy0.sh ${1}
sbatch --exclude=d1005 pmScripts/clip01_sum.sh ${1}
sbatch --exclude=d1005 pmScripts/clip01_ssim.sh ${1}
# phy
sbatch --exclude=d1005 pmScripts/clip01_autoW_sum_beta0.sh ${1}
sbatch --exclude=d1005 pmScripts/clip01_autoW_sum_beta2.sh ${1}
sbatch --exclude=d1005 pmScripts/clip01_autoW_sum_beta3.sh ${1}
sbatch --exclude=d1005 pmScripts/clip01_autoW_sum_beta10.sh ${1}
# n_stg
sbatch --exclude=d1005 pmScripts/clip01_autoW_sum_nStgT.sh ${1} 1
sbatch --exclude=d1005 pmScripts/clip01_autoW_sum_nStgT.sh ${1} 2
# sota mem, openpose, cycleGan, pix2pix