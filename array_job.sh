#!/bin/bash
##########################################
# Name the Job and set SBATCH parameters #
##########################################

#SBATCH --job-name=ARRAY_JOB
#SBATCH --array=0-29

#SBATCH -t 10
#SBATCH -n 50
#SBATCH -p short

######################
# Begin work section #
######################
set -e

job=$1
mydate=$(date +%d_%m_%y_%H%M)

echo "Sbatch job "$job" with task id "$SLURM_ARRAY_TASK_ID" was sent on "$mydate

./srun_apis.sh $job $SLURM_ARRAY_TASK_ID $SLURM_JOB_NAME
