#!/bin/bash
# FILENAME:  pk_cmass

#SBATCH --job-name=pk_cmass
#SBATCH --partition=pscomp
#SBATCH --exclusive
#SBATCH --time=5:00:00
#SBATCH --output=/data101/makinen/joblogs/pk_cmass_%A_%a.out    # Std-out (%A jobID, %a taskID)
#SBATCH --error=/data101/makinen/joblogs/pk_cmass_%A_%a.err     # Std-err
#SBATCH --mail-type=all       # Send email to above address at begin and end of job
#SBATCH --mem=2G
#SBATCH --array=0-6                    # <-- run tasks 0 … 9 (10 jobs)

# all of the other things here

module load cuda/12.1 # make this 12.1 ?
module load intelpython/3-2024.0.0

XLA_FLAGS=--xla_gpu_cuda_data_dir=\${CUDA_PATH}
export XLA_FLAGS


source /home/makinen/venvs/epe/bin/activate


# cd to top-level directory
cd /home/makinen/repositories/pk_cmass/


# --‐ Run the Python script, forwarding the array index ------------


## declare an array variable
declare -a arr=("hybrid" 
                "pk_bk_separate" 
                "pk_only_compression"
                "pk_bk_nocompress"
                "pk_only_nocompress"
                "bk_only_nocompress"
                )



python test_script.py --folder /data101/makinen/pk_cmass/4Lucas/fastpm_summaries/kmin-0.0_kmax-0.3/raw/ --experiment "${arr[${SLURM_ARRAY_TASK_ID}]}" --device cpu --test-folder /data101/makinen/pk_cmass/4Lucas/fastpm_summaries/kmin-0.0_kmax-0.3/raw/









