import shutil
import logging
import yaml
from typing import Any, Dict, Tuple
from peft import PeftModel, PeftMixedModel
from datasets import load_dataset
from scripts.modules.arguments import BaseArgs
from scripts.modules.data_module import get_tokenized_dataset
from scripts.modules.model_module import get_model_for_training
from scripts.utils.training_utils import get_training_args_kwargs
from scripts.utils.save_model_utils import save_model
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainArgs(BaseArgs):
    """
    Parser object inherited from BaseArgs class
    Get arguments from command-line
    """
    def __init__(self):
        super().__init__(description="Fine-tune ViT5 using QLoRa")
        self.add_lora_method()
        self.add_checkpoint_output_dirs()

    def add_lora_method(self):
        self.parser.add_argument(
            "--method",
            type=str,
            required=True,
            help="LORA / DORA/ ADALORA / OLORA"
        )

    def add_checkpoint_output_dirs(self):
        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            required=False,
            default="model_checkpoint",
            help="Where do you want to save the model's checkpoint?"
        )
    
def load_configs(configs_path: str) -> dict:
    """Load configs from YAML configuration file"""
    with open(configs_path, 'r', encoding='utf-8') as file:
        configs = yaml.safe_load(file)
    return configs

def load_tokenized_dataset(configs: dict) -> tuple:
    """Load tokenizer and preprocessing original dataset"""
    MODEL_NAME = configs["model"]["model_name"]
    DATASET_NAME = configs["model"]["dataset_name"]
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Load original dataset
    dataset = load_dataset(DATASET_NAME)
    # Tokenizing original dataset
    tokenized_dataset = get_tokenized_dataset(
        configs=configs,
        dataset=dataset,
        tokenizer=tokenizer
    )
    return tokenizer, tokenized_dataset

def load_model(configs: dict, method: str) -> PeftModel | PeftMixedModel:
    """
    Load base model with quantization technique
    Prepare model for training
    Combine model with PEFT settings
    Return model
    """
    model = get_model_for_training(
        configs=configs,
        method=method
    )
    return model

def load_trainer(
    configs: Dict[str, Any],
    tokenizer: AutoTokenizer,
    tokenized_dataset: Dict[str, Any],
    model: Any,
    checkpoint_dir: str
) -> Seq2SeqTrainer:
    """Initiate training args and trainer"""
    # Get training arguments kwargs
    training_args_kwargs = get_training_args_kwargs(configs=configs, checkpoint_dir=checkpoint_dir)

    # Feed the training arguments kwargs into Seq2Seq library
    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    # Use datacollator for padding 
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Use early stopping to prevent overfit
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5, 
        early_stopping_threshold=0.0
    )

    # Setup trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )

    return trainer

def main():
    # Print out running scripts
    terminal_width = shutil.get_terminal_size().columns
    logger.info(" RUNNING TRAIN.PY ".center(terminal_width, "="))

    # Load command-line arguments into this script
    args = TrainArgs().parse_args()

    # Load YAML configs
    configs = load_configs(args.configs)

    MODEL_NAME = configs["model"]["model_name"]

    logger.info(f" LOADING CONFIGURATION FOR {MODEL_NAME} FINE-TUNING COMPLETED ".center(terminal_width, "="))

    # Load tokenizer & preprocessing dataset
    tokenizer, tokenized_dataset = load_tokenized_dataset(configs=configs)

    logger.info(" PREPROCESSING DATA COMPLETED ".center(terminal_width, "="))

    # Initiate model for fine-tuning
    model = load_model(configs=configs, method=args.method)

    model.print_trainable_parameters()
    logger.info(f" STEP 3: SETTING UP {MODEL_NAME} QUANTIZATION COMPLETED ".center(terminal_width, "="))

    # Load trainer
    trainer = load_trainer(
        configs=configs,
        tokenizer=tokenizer,
        tokenized_dataset=tokenized_dataset,
        model=model,
        checkpoint_dir=args.checkpoint_dir
    )

    # Start Training
    logger.info(f" START TRAINING {MODEL_NAME}".center(terminal_width, "="))

    trainer.train()

    logger.info(f" TRAINING {MODEL_NAME} COMPLETED ".center(terminal_width, "="))

    # Save Model adapter & checkpoint
    save_model(
        trainer=trainer,
        tokenizer=tokenizer,
        adapter_dir=args.adapter_dir,
        checkpoint_dir=args.checkpoint_dir
    )

if __name__ == "__main__":
    main()