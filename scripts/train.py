import shutil
import yaml
from transformers import (
    AutoTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from scripts.modules.arguments import parse_args
from scripts.modules.data_module import prepare_dataset
from scripts.modules.quantization_module import model_quantization
from scripts.modules.lora_module import lora_configuration
from scripts.utils.save_model_module import save_model

def main():
    terminal_width = shutil.get_terminal_size().columns
    print(" RUNNING TRAIN.PY ".center(terminal_width, "="))

    args = parse_args()

    # 0. Read YAML Configs
    with open(args.configs, 'r', encoding='utf-8') as file:
        configs = yaml.safe_load(file)

    # ------------------------------------------------------------ TRAINING CONFIGS ------------------------------------------------------------
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

    # ------------------------------------------------------------ PREPROCESSING DATA ------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    tokenized_dataset = prepare_dataset(
        dataset_name=DATASET_NAME,
        tokenizer=tokenizer,
        train_samples=TRAIN_SAMPLES,
        val_samples=VAL_SAMPLES)
    
    print("STEP 2: PREPROCESSING DATA COMPLETED")

    # ------------------------------------------------------------ SETUP MODEL FOR TRAINING ------------------------------------------------------------
    base_model = model_quantization(model_name=MODEL_NAME)

    model = lora_configuration(
        base_model=base_model,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_target_module=LORA_TARGET_MODULE,
        lora_dropout=LORA_DROPOUT,
        use_dora=USE_DORA
    )

    model.print_trainable_parameters()

    print(F"STEP 3: SETTING UP {MODEL_NAME} QUANTIZATION COMPLETED")

    # ------------------------------------------------------------ SETUP HYPERPARAMETERS & TRAINER ------------------------------------------------------------

    # 4. Training Arguments
    training_args_kwargs = {
        "output_dir": CHECKPOINT_DIR,
        "num_train_epochs": EPOCHS,
        "learning_rate": LR,
        "lr_scheduler_type": "cosine",
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "label_smoothing_factor": LABEL_SMOOTHING_FACTOR,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "per_device_eval_batch_size": BATCH_SIZE * 2,
        "optim": "adamw_torch",
        "fp16": True,
        "logging_steps": 50,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "predict_with_generate": False,
        "report_to": "tensorboard",
        "disable_tqdm": True,
        "seed": 42
    }

    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)

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
    print(" START TRAINING ".center(terminal_width, "="))

    trainer.train()

    # 6. Save Model adapter
    save_model(
        trainer=trainer,
        tokenizer=tokenizer,
        adapter_dir=ADAPTER_DIR,
        checkpoint_dir=CHECKPOINT_DIR
    )

if __name__ == "__main__":
    main()