import shutil
import yaml
from scripts.modules.arguments import BaseArgs
from scripts.modules.quantization_module import model_quantization
from scripts.modules.lora_module import adalora_configuration
from scripts.utils.save_model_utils import save_model
from transformers import (
    AutoTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from scripts.utils.training_utils import (
    get_tokenized_dataset,
    get_total_steps,
    get_adalora_configs_kwargs,
    get_training_args_kwargs
)

class TrainArgs(BaseArgs):
    def __init__(self, description="Fine-tune ViT5 for Text Summarization"):
        super().__init__(description)

    def add_adapter_dir(self):
        self.parser.add_argument(
            
        )
    
def main():
    terminal_width = shutil.get_terminal_size().columns
    print(" RUNNING ADALORA_TRAIN.PY ".center(terminal_width, "="))

    args = parse_args()

    # 0. Read YAML Configs
    with open(args.configs, 'r', encoding='utf-8') as file:
        configs = yaml.safe_load(file)

    # ------------------------------------------------------------ TRAINING CONFIGS ------------------------------------------------------------
    # Model & Data    
    MODEL_NAME = configs["model"]["model_name"]

    # Output directory
    ADAPTER_DIR = args.adapter_dir
    CHECKPOINT_DIR = args.checkpoint_dir

    print(f"Step 0: Loading configuration for {MODEL_NAME} fine-tuning completed")

    # ------------------------------------------------------------ PREPROCESSING DATA ------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_dataset = get_tokenized_dataset(
        configs=configs, 
        tokenizer=tokenizer
    )
    
    print("STEP 2: PREPROCESSING DATA COMPLETED")

    # ------------------------------------------------------------ SETUP QUANTIZATION ADALORA MODEL FOR TRAINING ------------------------------------------------------------
    base_model = model_quantization(model_name=MODEL_NAME)

    total_steps = get_total_steps(
        configs=configs
    )

    adalora_configs_kwargs = get_adalora_configs_kwargs(
        configs=configs,
        total_steps=total_steps
    )
    
    model = adalora_configuration(
        base_model=base_model, 
        adalora_configs_kwargs=adalora_configs_kwargs
    )

    model.print_trainable_parameters()

    print(F"STEP 3: SETTING UP {MODEL_NAME} QUANTIZATION COMPLETED")

    # ------------------------------------------------------------ SETUP HYPERPARAMETERS & TRAINER ------------------------------------------------------------

    # 4. Training Arguments
    training_args_kwargs = get_training_args_kwargs(
        configs=configs, 
        checkpoint_dir=CHECKPOINT_DIR
    )
    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model
    )
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5, 
        early_stopping_threshold=0.0
    )

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