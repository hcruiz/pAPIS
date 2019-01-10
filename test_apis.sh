#!/bin/bash
set -e

STARTDIR=`pwd`
HCRTEMP=$(echo myTESTDIR)
#mkdir -p $HCRTEMP
cp -r Code/*.py ${HCRTEMP}
cp -r Data/ ${HCRTEMP}
cd ${HCRTEMP}
pwd

nodes=$1
job=$2
job_nr=$3

filename="TEST_APIS_"${job%.*}"_n"$nodes
mydate=$(date +%d_%m_%y_%H%M%S)

echo "Testing job "$filename" with ID "$mydate 
mpiexec -n $nodes python $job $job_nr $4 $filename $5 | tee $filename.pyout

dir=$(echo $STARTDIR/TEST_dump)

mv $filename*.* $dir/

cd $STARTDIR
echo "done"
