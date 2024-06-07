#!/bin/bash

## See <https://slurm.schedmd.com/sbatch.html> para mais informacoes sobre

#SBATCH --partition=fast 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 #16
#SBATCH --time=1:00:00 #10
#SBATCH --job-name=qcnn_train

# Job settings

INP=$SLURM_JOB_NAME".toml"
OUT=$SLURM_JOB_NAME".out"
ERR=$SLURM_JOB_NAME".err"
JOB_DIR=$SLURM_JOB_NAME

# Job env

export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

source $HOME/QCNN-Metrics/venv/bin/activate

mkdir $JOB_DIR
cd $JOB_DIR

# Job

echo -e "Starting job $(date +'%d-%m-%Y as %T')"
echo -e "$USER active jobs:"
squeue -a --user=$USER
echo -e "Job node: $(hostname -s)"
echo -e "Job tasks: $SLURM_NTASKS"
echo -e "Job threads: $OMP_NUM_THREADS"
echo -e "Submit directory: $SLURM_SUBMIT_DIR"
echo -e "Scratch directory: $WRKDIR"
echo -e "Input file: $INP"

python $HOME/QCNN-Metrics/src/qcnn.py $INP 1> $OUT 2> $ERR
deactivate

echo -e "Ending job $(date +'%d-%m-%Y as %T')"

# Post-processing

echo -e "Post processing"

cp $JOB_DIR $HOME
rm -rf $JOB_DIR

echo -e "Done $(date +'%d-%m-%Y as %T')"
