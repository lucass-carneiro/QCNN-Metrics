export OMP_NUM_THREADS=18
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

source venv/bin/activate

echo $OMP_NUM_THREADS
echo $OMP_PROC_BIND
echo $OMP_PLACES

python src/qcnn.py fit.toml
