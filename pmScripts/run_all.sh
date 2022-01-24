#!/bin/bash
sbatch pmScripts/clip01.sh ${1}
sbatch pmScripts/clip01_autoW.sh ${1}
sbatch pmScripts/clip01_pwrs0.sh ${1}
sbatch pmScripts/clip01_autoW_sum.sh ${1}
sbatch pmScripts/clip01_autoW_sum_ssim.sh ${1}
sbatch pmScripts/clip01_autoW_sum_ssim_D.sh ${1}
sbatch pmScripts/clip01_D.sh ${1}
#sbatch pmScripts/clip01_nPhy0.sh
sbatch pmScripts/clip01_sum.sh ${1}
sbatch pmScripts/clip01_ssim.sh ${1}
#sbatch pmScripts/openPose.sh
# phy
sbatch pmScripts/clip01_autoW_sum_beta2.sh ${1}
sbatch pmScripts/clip01_autoW_sum_beta3.sh ${1}
sbatch pmScripts/clip01_autoW_sum_beta10.sh ${1}
