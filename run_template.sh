#!/bin/bash

# Dừng script ngay lập tức nếu có bất kỳ lỗi nào xảy ra
set -e 

echo "=== SETTING UP EXPERIMENT ==="

# --- IMPORTANT PATHS ---
WORKING_BRANCH="hp_search"
TEMPLATE_PATH="configs/baseline/training_configs.yaml.template"
CONFIG_PATH="configs/baseline/training_configs.yaml"
SCRIPT="scripts/baseline/qdora_vit5/train.py"                       
OUTPUT_DIR="model_adapter"
# --------------------------------

export LEARNING_RATE=0.0003143144372398535

# clone repo
git clone https://github.com/VietAnh-AI2-UET/ViT5_Base_QLoRa_VNDS.git

# Move to working directory
cd ViT5_Base_QLoRa_VNDS

# checkout current working branch
git checkout $WORKING_BRANCH

# install gettext for bash
apt-get install gettext

# install requirements.txt
pip install requirements.txt

# Sinh file cấu hình hoàn chỉnh từ file yaml template
echo "1. Creating configs file at: $CONFIG_PATH"

(echo "cat <<EOF"; cat $TEMPLATE_PATH; echo ""; echo "EOF") | bash > $CONFIG_PATH

# Bắt đầu chạy file
echo "2. Start running script"

python $SCRIPT --config $CONFIG_PATH --output_dir $OUTPUT_DIR

echo "=== FINISH ==="