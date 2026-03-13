import os
import shutil
import yaml
import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ViT5 using QDoRa for Text Summarization")
    
    parser.add_argument(
        "--configs", 
        type=str,
        required=True, # Bắt buộc phải truyền file config khi chạy
        help="Enter YAML training configs file path"
    )

    parser.add_argument(
        "--use_dora",
        type=bool,
        required=True,
        help="Which methode do you want to use? DORA --> True / LORA --> False"
    )

    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=False,
        default="model_adapter",
        help="Where do you want to save the model's adapter?"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
        default="model_checkpoint",
        help="Where do you want to save the model's checkpoint?"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    print("===== RUNNING TRAIN.PY =====")

    # 0. Read YAML Configs
    with open(args.configs, 'r', encoding='utf-8') as file:
        configs = yaml.safe_load(file)

    # --- TRAINING PARAMETERS ---
    # Model & Data    
    MODEL_NAME = configs["model"]["model_name"]
    DATASET_NAME = configs["model"]["dataset_name"]

    # Data sampling
    TRAIN_SAMPLES = configs["data"]["train_samples"]
    VAL_SAMPLES = configs["data"]["val_samples"]

    # LoRa configs
    LORA_R = configs["lora"]["lora_r"]
    LORA_ALPHA = configs["lora"]["lora_alpha"]
    LORA_TARGET_MODULE = configs["lora"]["lora_target_module"]
    LORA_DROPOUT = configs["lora"]["lora_dropout"]
    USE_DORA = args.use_dora

    # Training Hyperparams
    EPOCHS = configs["training"]["epochs"]
    LR = configs["training"]["learning_rate"]
    WEIGHT_DECAY = configs["training"]["weight_decay"]
    WARMUP_RATIO = configs["training"]["warmup_ratio"]
    LABEL_SMOOTHING_FACTOR = configs["training"]["label_smoothing_factor"]
    BATCH_SIZE = configs['training']['batch_size']
    GRAD_ACCUM_STEPS = configs['training']['gradient_accumulation_steps']

    # Output directory
    ADAPTER_DIR = args.adapter_dir
    CHECKPOINT_DIR = args.checkpoint_dir

    print(f"Step 0: Loading configuration for {MODEL_NAME} fine-tuning completed")
    # --------------------------------------------------------------------------------

    # 1. Load Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset(DATASET_NAME)

    # Subsample dataset
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))                   # We only need about 6000 samples for fine-tuning
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(VAL_SAMPLES))           # And only 1000 samples for validation

    print("STEP 1: LOADING TOKENIZER AND DATASET COMPLETED")
    
    # 2. Preprocess Data
    def preprocess_function(batch):
        input_texts = ['tóm tắt: ' + txt for txt in batch['article']]
        input_max_length = 1024
        target_max_length = 256

        model_inputs = tokenizer(input_texts, max_length=input_max_length, truncation=True)
        labels = tokenizer(text_target=batch['abstract'], max_length=target_max_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    print("STEP 2: PREPROCESSING DATA COMPLETED")

    # 3. Setup QLORA/QDORA Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map='auto'
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULE,
        lora_dropout=LORA_DROPOUT,
        bias='none',
        task_type='SEQ_2_SEQ_LM',
        use_dora=USE_DORA
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()      # <--- QLORA/QDORA VIT5-BASE (baseline model)

    print(F"STEP 3: SETTING UP {MODEL_NAME} QUANTIZATION COMPLETED")

    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,      # This is where "report_to="tensorboard"" folder is stored
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        label_smoothing_factor=LABEL_SMOOTHING_FACTOR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        optim="adamw_torch",
        fp16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        report_to="tensorboard",        # Report training logs to tensorboard
        seed=42
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)
    
    print("STEP 4: SETTING UP HYPERPARAMS COMPLETED")

    # 5. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )

    # Start Training
    print(f"===== START TRAINING {MODEL_NAME} =====")
    trainer.train()

    # 6. Save Model adapter
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    trainer.save_model(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    print(f"Training completed! Model saved at: {ADAPTER_DIR}")

    # Create archive file for model's adapter
    shutil.make_archive(ADAPTER_DIR, "zip", ADAPTER_DIR)

    print(f"Thư mục '{ADAPTER_DIR}' đã được nén thành công thành '{ADAPTER_DIR}.zip'")
    print(f"Bạn có thể tải tệp '{ADAPTER_DIR}.zip' từ phần duyệt tệp của Colab (biểu tượng thư mục ở bên trái).")

    # Create archive file for model's checkpoints
    shutil.make_archive(CHECKPOINT_DIR, "zip", CHECKPOINT_DIR)

    print(f"Thư mục '{CHECKPOINT_DIR}' đã được nén thành công thành '{CHECKPOINT_DIR}.zip'")
    print(f"Bạn có thể tải tệp '{CHECKPOINT_DIR}.zip' từ phần duyệt tệp của Colab (biểu tượng thư mục ở bên trái).")

if __name__ == "__main__":
    main()