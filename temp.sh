export OMP_NUM_THREADS=18
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

source venv/bin/activate

echo "Using" $OMP_NUM_THREADS "threads"
echo "Process binding:" $OMP_PROC_BIND
echo "Places:" $OMP_PLACES

python src/qcnn.py fit.toml