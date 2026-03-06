import os
import yaml
import argparse
import torch
import numpy as np
import optuna
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
    parser = argparse.ArgumentParser(description="Hyperparameter Search for QDoRa ViT5 using Optuna")
    
    parser.add_argument(
        "--config", 
        type=str,
        required=True,
        help="Enter YAML configs path"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where do you want to save the model's adapter?"
    )
    
    # The number of trials for Optuna hp search
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of trials for Optuna hyperparameter search"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    print("Start running train.py with Optuna Integration")

    # 0. Read YAML Configs
    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Model & Data
    MODEL_NAME = config["model"]["model_name"]
    DATASET_NAME = config["model"]["dataset_name"]

    # Data sampling
    TRAIN_SAMPLES = config["data"]["train_samples"]
    VAL_SAMPLES = config["data"]["val_samples"]

    # LoRa configs
    LORA_R = config["lora"]["lora_r"]
    LORA_ALPHA = config["lora"]["lora_alpha"]
    LORA_TARGET_MODULE = config["lora"]["lora_target_module"]
    LORA_DROPOUT = config["lora"]["lora_dropout"]

    # Training Hyperparams
    EPOCHS = config["training"]["epochs"]
    # LR = config["training"]["learning_rate"]          # Learning rate will be for Optuna to decide
    WEIGHT_DECAY = config["training"]["weight_decay"]
    WARMUP_RATIO = config["training"]["warmup_ratio"]
    LABEL_SMOOTHING_FACTOR = config["training"]["label_smoothing_factor"]
    BATCH_SIZE = config['training']['batch_size']
    GRAD_ACCUM_STEPS = config['training']['gradient_accumulation_steps']

    OUTPUT_DIR = args.output_dir

    print(f"Step 0: Loading configuration for {MODEL_NAME} completed")

    # 1. Load Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset(DATASET_NAME)

    # Subsample dataset
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))               # We should only use 1k samples for hp search
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(VAL_SAMPLES))       # And validation should only be 200

    print("Step 1: Loading tokenizer and dataset completed")
    
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
    
    print("Step 2: Preprocessing dataset completed")

    # 3. In hp search process, we must initiate a new model after each trial
    def model_init():
        """Return a brand new model for each trial"""
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
            use_dora=True
        )
        
        return get_peft_model(base_model, lora_config)

    print(f"Step 3: Setting up model_init wrapper for QDoRa completed")

    # 4. THIẾT LẬP OPTUNA
    def optuna_hp_space(trial):
        """--Hyperparameters search space--
            Define which parameter you want to perform search with Optuna"""
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        }

    def compute_objective(metrics):
        """
        Return the target that you want to optimize (currently validation loss)
        """
        return metrics["eval_loss"]

    # 5. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./qdora_vit5_search_checkpoints",
        num_train_epochs=EPOCHS,
        # learning_rate will be overwritten by Optuna
        # learning_rate=LR
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
        save_total_limit=1,                     # This script is just for searching hp phase, so actually no need for any saving 
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        report_to="tensorboard",
        seed=42
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model_init=model_init)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)
    
    print("Step 4: Setting up Training hyperparams completed")

    # 6. Trainer
    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )

    # 7. RUN OPTUNA HYPERPARAMETER SEARCH
    print(f"Step 5: Start hyperparameters search with {args.n_trials} trials...")
    best_trial = trainer.hyperparameter_search(
        direction="minimize", # Depend on your compute objective (in this case validation loss, so it should be minimize)
        backend="optuna",
        hp_space=optuna_hp_space,
        compute_objective=compute_objective,
        n_trials=args.n_trials
    )

    print("="*50)
    print("Hyperparameters searching completed!")
    print(f"Best hp found: {best_trial.hyperparameters}")
    print(f"Best eval_loss: {best_trial.objective}")
    print("="*50)

if __name__ == "__main__":
    main()