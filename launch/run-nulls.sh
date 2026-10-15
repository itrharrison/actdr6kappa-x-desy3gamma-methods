#!/bin/bash --login

###
#job name
#SBATCH --job-name=null_examples
#job stdout file
#SBATCH --output=null_examples.out.%J
#job stderr file
#SBATCH --error=null_examples.err.%J
#maximum job time in D-HH:MM
#SBATCH --time=0-72:00
#number of parallel processes (tasks) you are requesting - maps to MPI processes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=191000
#SBATCH -p c_compute_physics1
#SBATCH -A scw1361
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=harrisoni@cardiff.ac.uk
###

module load Miniforge3

source activate
conda deactivate

export CONDA_ENVS_PATH=/home/scw1361/.conda/envs
export ACTDIR=/scratch/c.c1025819/actdr6_nulls

conda activate /scratch/c.c1025819/envs/actxdes

export OMP_NUM_THREADS=8

which python >> $SLURM_SUBMIT_DIR/pyver.txt
#python $SLURM_SUBMIT_DIR/scripts/example_null_test.py
mpirun -n 4 python $SLURM_SUBMIT_DIR/scripts/get_sim_covmats.py
