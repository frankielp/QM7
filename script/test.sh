#!/bin/bash
#SBATCH --job-name=valml        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=3G         # memory per cpu-core (4G is default)
#SBATCH --time=10-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=<YourNetID>@princeton.edu
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
####SBATCH --nodelist=selab4
#SBATCH -otest.out
#SBATCH -etest.err

module purge
module load anaconda3-2021.05-gcc-9.3.0-r6itwa7
# export PATH=/usr/local/cuda-11.5/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/

nvidia-smi
nvcc --version

## ENV ##
conda init bash 
source activate qm7
## TRAIN ##
cd ..
python train.py
## PREDICT

## TEST

conda deactivate



