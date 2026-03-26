import shutil
import logging
import yaml
from typing import Any, Dict, Tuple
from peft import PeftModel, PeftMixedModel
from datasets import load_dataset
from .modules.arguments import BaseArgs
from .modules.data_module import get_tokenized_dataset
from .modules.model_module import get_model_for_training
from .utils.training_utils import get_training_args_kwargs
from .utils.save_model_utils import save_model
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
        self.add_number_of_trials()
        self.add_checkpoint_output()

    def add_lora_method(self):
        self.parser.add_argument(
            "--method",
            type=str,
            required=True,
            help="LORA / DORA/ ADALORA / OLORA"
        )
    def add_number_of_trials(self):
        self.parser.add_argument(
            "--n_trials",
            type=int,
            required=False,
            default=5,
            help="Enter number of trials to perform search"
        )
    def add_checkpoint_output(self):
        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            required=False,
            default="model_checkpoint",
            help="Enter model's checkpoint saving location"
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

def hp_space(trials: Any) -> Dict[str, Any]:
    """Define which parameter you want to perform search with Optuna"""
    return {
        "learning_rate": trials.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
    }

def compute_objective(metrics: Dict[str, Any]) -> float:
    """Return the target that you want to optimize (currently validation loss)"""
    return metrics["eval_loss"]

def load_trainer(
    configs: Dict[str, Any],
    tokenizer: AutoTokenizer,
    tokenized_dataset: Dict[str, Any],
    method: str,
    checkpoint_dir: str
) -> Seq2SeqTrainer:
    """Initiate training args and trainer"""
    # Get training arguments kwargs
    training_args_kwargs = get_training_args_kwargs(configs=configs, checkpoint_dir=checkpoint_dir)

    # Feed the training arguments kwargs into Seq2Seq library
    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    # Create dummy model for data collator and trainer
    dummy_model = load_model(
        configs=configs,
        method=method
    )
    # Use datacollator for padding 
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=dummy_model)

    # Use early stopping to prevent overfit
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5, 
        early_stopping_threshold=0.0
    )

    # Setup trainer
    trainer = Seq2SeqTrainer(
        model=dummy_model,
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
    logger.info(" RUNNING HP_SEARCH.PY ".center(terminal_width, "="))

    # Load command-line arguments into this script
    args = TrainArgs().parse_args()

    # Load YAML configs
    configs = load_configs(args.configs)

    MODEL_NAME = configs["model"]["model_name"]
    CHECKPOINT_DIR = args.checkpoint_dir

    logger.info(f" LOADING CONFIGURATION FOR {MODEL_NAME} FINE-TUNING COMPLETED ".center(terminal_width, "="))

    # Load tokenizer & preprocessing dataset
    tokenizer, tokenized_dataset = load_tokenized_dataset(configs=configs)

    logger.info(" PREPROCESSING DATA COMPLETED ".center(terminal_width, "="))

    # Load trainer
    trainer = load_trainer(
        configs=configs,
        tokenizer=tokenizer,
        tokenized_dataset=tokenized_dataset,
        method=args.method,
        checkpoint_dir=args.checkpoint_dir
    )

    # Start Training
    logger.info(f" START TRAINING {MODEL_NAME}".center(terminal_width, "="))

    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        backend="optuna",
        hp_space=hp_space,
        compute_objective=compute_objective,
        n_trials=args.n_trials
    )

    logger.info(f" TRAINING {MODEL_NAME} COMPLETED ".center(terminal_width, "="))

    print("="*50)
    print("Hyperparameters searching completed!")
    print(f"Best hp found: {best_trial.hyperparameters}")
    print(f"Best eval_loss: {best_trial.objective}")
    print("="*50)

    shutil.make_archive(CHECKPOINT_DIR, "zip", CHECKPOINT_DIR)

    print(f"Thư mục '{CHECKPOINT_DIR}' đã được nén thành công thành '{CHECKPOINT_DIR}.zip'")
          
if __name__ == "__main__":
    main()