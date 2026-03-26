#!/bin/bash

# Dừng script ngay lập tức nếu có bất kỳ lỗi nào xảy ra
set -e 

echo "=== SETTING UP INFERENCING ==="

# --- IMPORTANT PATHS ---
TEMPLATE_PATH="enter the .yaml.template file path"
CONFIG_PATH="that .yaml.template file will create the real .yaml here"
INFER_SCRIPT="enter the infering script that you want to run"                     
ADAPTER_PATH="where did you save the model's adapter?"
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

# Start inferencing
echo "Start running $INFER_SCRIPT"

python -m $INFER_SCRIPT --config $CONFIG_PATH --adapter_path $ADAPTER_PATH

echo "=== FINISH ==="