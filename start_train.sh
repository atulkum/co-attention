source activate squad
export PYTHONPATH=`pwd`/code
python code/co-attention/process_training.py train >& log/train_log &
