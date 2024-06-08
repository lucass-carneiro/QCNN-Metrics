#!/bin/bash

## See <https://slurm.schedmd.com/sbatch.html> para mais informacoes sobre

#SBATCH --partition=fast 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 #16
#SBATCH --time=1:00:00 #10
#SBATCH --job-name=qcnn_train

# Job settings

PAR_FILE=$HOME/$SLURM_JOB_NAME".toml"
OUT_FILE=$SLURM_JOB_NAME".out"
ERR_FILE=$SLURM_JOB_NAME".err"

JOB_DIR=$SLURM_JOB_NAME

QCNN_DIR=$HOME/QCNN-Metrics
QCNN_REQUIREMENTS=$QCNN_DIR/requirements.txt
QCNN_MAIN=$QCNN_DIR/src/qcnn.py
QCNN_SCRIPT=$QCNN_DIR/sing/qcnn.sh

QCNN_SING_IMG=$HOME/qcnn.simg

# Job env

module load singularity

export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Job dirs

mkdir $JOB_DIR
cd $JOB_DIR

# Job

echo -e "Starting job $(date +'%d-%m-%Y as %T')"
echo -e "Job node: $(hostname -s)"
echo -e "Job tasks: $SLURM_NTASKS"
echo -e "Job threads: $OMP_NUM_THREADS"
echo -e "Submit directory: $SLURM_SUBMIT_DIR"

singularity exec $QCNN_SING_IMG bash $QCNN_SCRIPT $QCNN_REQUIREMENTS $QCNN_MAIN $PAR_FILE 1> $OUT_FILE 2> $ERR_FILE

echo -e "Ending job $(date +'%d-%m-%Y as %T')"
