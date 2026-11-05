#!/bin/bash
# Runs the Fortran code at h = 2, 1, 0.5, 0.25 and saves each run to data_h=<h>/
# Put this next to tdgl.F90. Run with:  bash run_fortran.sh

COMPILE="mpifort -fopenmp -I/usr/include tdgl.F90 -o tdgl -ldmumps -lscalapack-openmpi -lopenblas"
RUN="./tdgl"

export OMP_NUM_THREADS=1

for pair in "2.0 16" "1.0 31" "0.5 61" "0.25 121"; do
    set -- $pair
    h=$1
    d=$2
    echo "==== h = $h   dimx = dimy = $d ===="

    # set grid size in the source
    sed -i.bak "s/parameter :: dimx = [0-9]*/parameter :: dimx = $d/" tdgl.F90
    sed -i.bak "s/parameter :: dimy = [0-9]*/parameter :: dimy = $d/" tdgl.F90

    $COMPILE || exit 1

    rm -rf data
    $RUN | tee log.txt

    rm -rf "data_h=$h"
    mkdir -p "data_h=$h"
    mv data/* "data_h=$h/"
    mv log.txt "data_h=$h/"

    # save total wall time (last number on the "total runtime" line)
    grep "total runtime" "data_h=$h/log.txt" | awk '{print $NF}' > "data_h=$h/time.txt"
done

rm -f tdgl.F90.bak
echo "done"