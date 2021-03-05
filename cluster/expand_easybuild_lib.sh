#! /bin/bash
module purge
module load GCC
module load OpenMPI
module load $1
path=$(python -c "import $2; print($2.__file__)")
echo $path 
