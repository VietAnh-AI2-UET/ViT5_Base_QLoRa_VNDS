#!/bin/bash

# Dừng script ngay lập tức nếu có bất kỳ lỗi nào xảy ra
set -e 

echo "=== SETTING UP TRAINING EXPERIMENT ==="

# --- IMPORTANT PATHS ---
TEMPLATE_PATH="enter the .yaml.template file path"
CONFIG_PATH="that .yaml.template file will create the real .yaml here"
TRAIN_SCRIPT="enter the training script that you want to run (relative path from cwd)."
METHOD="LORA / DORA / ADALORA / OLORA"                       
ADAPTER_DIR="where do you want to save the model's adapter?"
CHECKPOINT_DIR="where do you want to save the model's checkpoint?"
# --------------------------------

# --- HYPERPARAMS ADJUSTMENT ---
# export LEARNING_RATE=0.0003143144372398535
# --------------------------------

# install gettext for bash
echo "Installing gettext library for bash"
apt-get install gettext

# install requirements.txt
echo "Installing requirements.txt"
pip install -qq -r requirements.txt

# Generate YAML config file from YAML.TEMPLATE
echo "Creating configs file at: $CONFIG_PATH"

(echo "cat <<EOF"; cat $TEMPLATE_PATH; echo ""; echo "EOF") | bash > $CONFIG_PATH

# Start training
echo "Start running $TRAIN_SCRIPT"

python -m $TRAIN_SCRIPT --config $CONFIG_PATH --method $METHOD --adapter_dir $ADAPTER_DIR --checkpoint_dir $CHECKPOINT_DIR

echo "=== FINISH ==="