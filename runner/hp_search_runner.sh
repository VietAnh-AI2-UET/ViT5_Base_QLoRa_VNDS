#!/bin/bash

# Dừng script ngay lập tức nếu có bất kỳ lỗi nào xảy ra
set -e 

echo "=== SETTING UP HYPERPARAMS SEARCHING EXPERIMENT ==="

# --- IMPORTANT PATHS ---
TEMPLATE_PATH="enter the .yaml.template file path"
CONFIG_PATH="that .yaml.template file will create the real .yaml here"
TRAIN_SCRIPT="enter the training script that you want to run"
N_TRIALS=5
METHOD="LORA / DORA / ADALORA / OLORA"                       
CHECKPOINT_DIR="where do you want to save the model's checkpoint?"

# --- HYPERPARAMS ADJUSTMENT ---
# export LEARNING_RATE=0.0003143144372398535
# --------------------------------

# install gettext for bash
echo "Installing gettext library for bash"
apt-get install gettext

# install requirements.txt
echo "Installing requirements.txt"
pip install -qq -r requirements.txt

# Sinh file cấu hình hoàn chỉnh từ file yaml template
echo "Creating configs file at: $CONFIG_PATH"

(echo "cat <<EOF"; cat $TEMPLATE_PATH; echo ""; echo "EOF") | bash > $CONFIG_PATH

# Start training
echo "Start running $TRAIN_SCRIPT"

# Delete the --use_dora parameter if you want to use AdaLORA
python -m $TRAIN_SCRIPT --config $CONFIG_PATH --n_trials $N_TRIALS --method $METHOD --checkpoint_dir $CHECKPOINT_DIR

echo "=== FINISH ==="