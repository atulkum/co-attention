source activate squad
export PYTHONPATH=`pwd`/code
python code/process_training.py train >& log/train_log &
