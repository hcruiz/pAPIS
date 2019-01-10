#!/bin/bash
module load python/2.7.9
set -e
STARTDIR=`pwd`
HCRTEMP=`mktemp -d ${TMPDIR}/hcrtemp.XXXXX`
cp -r Data/ ${HCRTEMP}
cp -r Code/* ${HCRTEMP}
cd ${HCRTEMP}
pwd

job=$1
job_nr=$2
main_jobdir=$3
mydate=$4

jobid=${job%.*}
jobid=${jobid#Main_}
casedir=$jobid"_Nr"$job_nr

fileprefix="APIS_n"$SLURM_NPROCS

echo "Processing job "$job" with ID "$casedir" using "$SLURM_NPROCS" nodes"
srun python $job $job_nr -save $fileprefix | tee ${fileprefix}.pyout

dir=$(echo $STARTDIR"/Results/Output/"$main_jobdir"/"$SLURM_ARRAY_JOB_ID"/"$casedir)

mkdir -p $dir/
mv *$jobid*.py $dir/
mv $fileprefix*.* $dir/
echo "All files placed in "$dir/
cd $STARTDIR
mv slurm-${SLURM_ARRAY_JOB_ID}_${job_nr}.out slurm_out
echo "done"
