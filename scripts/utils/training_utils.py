import math
from scripts.modules.data_module import prepare_dataset

def get_tokenized_dataset(configs, tokenizer):
    tokenized_dataset = prepare_dataset(
        dataset_name=configs["model"]["dataset_name"],
        tokenizer=tokenizer,
        train_samples=configs["data"]["train_samples"],
        val_samples=configs["data"]["val_samples"]
    )
    return tokenized_dataset

def get_total_steps(configs):
    TRAIN_SAMPLES = configs["data"]["train_samples"]
    BATCH_SIZE = configs["training"]["batch_size"]
    GRAD_ACCUM_STEPS = configs['training']['gradient_accumulation_steps']
    EPOCHS = configs["training"]["epochs"]

    steps_per_epoch = math.ceil(TRAIN_SAMPLES / (BATCH_SIZE * GRAD_ACCUM_STEPS))
    total_steps = steps_per_epoch * EPOCHS
    
    return total_steps

def get_lora_configs_kwargs(configs, use_dora):
    lora_configs_kwargs = {
        "r": configs["lora"]["lora_r"],
        "lora_alpha": configs["lora"]["lora_alpha"],
        "lora_dropout": configs["lora"]["lora_target_module"],
        "target_modules": configs["lora"]["lora_dropout"],
        "bias": "none",
        "task_type": "SEQ_2_SEQ_LM",
        "use_dora": use_dora
    }
    return lora_configs_kwargs

def get_adalora_configs_kwargs(configs, total_steps):
    adalora_configs_kwargs = {
        "init_r": configs["lora"]["init_r"],
        "target_r": configs["lora"]["target_r"],
        "tinit": configs["lora"]["tinit"],
        "tfinal": configs["lora"]["tfinal"],
        "deltaT": configs["lora"]["delta_t"],
        "total_step": total_steps,
        "lora_alpha": configs["lora"]["lora_alpha"],
        "target_modules": configs["lora"]["lora_target_module"],
        "lora_dropout": configs["lora"]["lora_dropout"],
        "bias": 'none',
        "task_type": 'SEQ_2_SEQ_LM'
    }

    return adalora_configs_kwargs

def get_training_args_kwargs(configs, checkpoint_dir):
    """Build training arguments kwargs from configs."""
    return {
        "output_dir": checkpoint_dir,
        "num_train_epochs": configs["training"]["epochs"],
        "learning_rate": configs["training"]["learning_rate"],
        "lr_scheduler_type": "cosine",
        "weight_decay": configs["training"]["weight_decay"],
        "warmup_ratio": configs["training"]["warmup_ratio"],
        "label_smoothing_factor": configs["training"]["label_smoothing_factor"],
        "per_device_train_batch_size": configs["training"]["batch_size"],
        "gradient_accumulation_steps": configs['training']['gradient_accumulation_steps'],
        "per_device_eval_batch_size": configs["training"]["batch_size"] * 2,
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