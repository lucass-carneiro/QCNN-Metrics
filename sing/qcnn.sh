python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --upgrade -r $1

python $2 $3

deactivate