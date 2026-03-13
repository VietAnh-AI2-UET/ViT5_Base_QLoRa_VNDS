#!/bin/bash

# Dừng script ngay lập tức nếu có bất kỳ lỗi nào xảy ra
set -e 

echo "=== SETTING UP TRAINING EXPERIMENT ==="

# --- IMPORTANT PATHS ---
TEMPLATE_PATH="enter configs template file path"
CONFIG_PATH="where do you want to save the yaml file?"
TRAIN_SCRIPT="enter training script path"
USE_DORA="Which methode do you want to use? DORA --> True / LORA --> False"                       
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

# Sinh file cấu hình hoàn chỉnh từ file yaml template
echo "Creating configs file at: $CONFIG_PATH"

(echo "cat <<EOF"; cat $TEMPLATE_PATH; echo ""; echo "EOF") | bash > $CONFIG_PATH

# Start training
echo "Start running $TRAIN_SCRIPT"

python $TRAIN_SCRIPT --config $CONFIG_PATH --use_dora $USE_DORA --adapter_dir $ADAPTER_DIR --checkpoint_dir $CHECKPOINT_DIR

echo "=== FINISH ==="