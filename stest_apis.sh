#!/bin/bash
module load python/2.7.9
set -e
##cd Code
STARTDIR=`pwd`
HCRTEMP=`mktemp -d ${TMPDIR}/hcrtemp.XXXXX`
cp -r Data/ ${HCRTEMP}
cp -r Code/* ${HCRTEMP}
cd ${HCRTEMP}
pwd

nodes=$1
job=$2
job_nr=$3

filename="TEST_APIS_"${job%.*}"_n"$nodes
mydate=$(date +%d_%m_%y_%H%M%S)

echo "Testing job "$filename" with ID "$mydate 
srun -p short -t 15 -n $nodes python $job $job_nr $4 $filename | tee $filename.pyout

dir=$(echo $STARTDIR/TEST_dump)

mv $filename*.* $dir/

echo "done"
