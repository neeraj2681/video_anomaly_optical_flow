#!bash/bin/sh

pip install -r requirements.txt
python3 gpu_check.py
python3 create_training_set.py
python3 model.py
python3 model_summary.py

