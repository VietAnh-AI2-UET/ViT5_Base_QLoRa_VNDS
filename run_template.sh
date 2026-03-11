#!/bin/bash

# Dừng script ngay lập tức nếu có bất kỳ lỗi nào xảy ra
set -e 

echo "=== SETTING UP EXPERIMENT ==="

# --- IMPORTANT PATHS ---
TEMPLATE_PATH="configs/baseline/training_configs.yaml.template"
CONFIG_PATH="configs/baseline/training_configs.yaml"
SCRIPT="scripts/baseline/qdora_vit5/train.py"                       
OUTPUT_DIR="model_adapter"
# --------------------------------

# export LEARNING_RATE=0.0003143144372398535

# install gettext for bash
echo "installing gettext library for bash"
apt-get install gettext

# install requirements.txt
echo "installing requirements.txt"
pip install -qq -r requirements.txt

# Sinh file cấu hình hoàn chỉnh từ file yaml template
echo "Creating configs file at: $CONFIG_PATH"

(echo "cat <<EOF"; cat $TEMPLATE_PATH; echo ""; echo "EOF") | bash > $CONFIG_PATH

# Bắt đầu chạy file
echo "Start running script"

python $SCRIPT --config $CONFIG_PATH --output_dir $OUTPUT_DIR

echo "=== FINISH ==="