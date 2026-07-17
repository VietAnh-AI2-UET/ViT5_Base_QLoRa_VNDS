import shutil
import yaml
from typing import Any, Dict
from peft import PeftModel, PeftMixedModel
from datasets import load_dataset
from .modules.parse_module import BaseArgs
from .modules.data_module import get_tokenized_dataset
from .modules.model_module import get_model_for_training
from .utils.args_utils import get_training_args_kwargs
from .utils.save_utils import save_model
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

class TrainArgs(BaseArgs):
    """
    Parser object inherited from BaseArgs class
    Get arguments from command-line
    """
    def __init__(self):
        super().__init__(description="Fine-tune ViT5 using QLoRa")
        self.add_lora_method()
        self.add_adapter_output()
        self.add_checkpoint_output()

    def add_lora_method(self):
        # this argument specify which lora method will be use for training: 
        # normal lora or asymetric lora
        # let name it: "method"
        self.parser.add_argument(
            "--method",
            type=str,
            required=True,
            help="normal / asym"
        )

    def add_adapter_output(self):
        # this argument specify where will the trained model's adapter file will be stored
        # let name it: "adapter_dir"
        self.parser.add_argument(
            "--adapter_dir",
            type=str,
            required=False,
            default="model_adapter",
            help="Enter model's adapter saving location"
        )

    def add_checkpoint_output(self):
        # this argument specify where will the trained model's checkpoint file will be stored
        # let name it: "checkpoint_dir"
        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            required=False,
            default="model_checkpoint",
            help="Enter model's checkpoint saving location"
        )
    
def load_configs(configs_path: str) -> dict:
    """Load configs from YAML configuration file"""
    # in this function, we will open the path to YAML config file being passed by terminal command-line
    # after that, we will read and save all the parameters inside the variable named "configs" and return
    with open(configs_path, 'r', encoding='utf-8') as file:
        configs = yaml.safe_load(file)
    return configs

def load_tokenized_dataset(tokenizer, dataset, configs) -> tuple:
    """Preprocessing original dataset"""
    
    # Tokenizing original dataset
    tokenized_dataset = get_tokenized_dataset(
        configs=configs,
        dataset=dataset,
        tokenizer=tokenizer
    )

    return tokenized_dataset

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
    print(" RUNNING TRAIN.PY ".center(terminal_width, "="))

    # First of all, we have to read all the arguments 
    # that we pass into this file through termial cmd
    args = TrainArgs().parse()

    # The first argument we take from command-line is path to YAML config
    # let this variable be: "configs"
    configs = load_configs(args.configs)

    # The second argument we take from command-line is 
    # the one that specify which LORA method will be use
    # we'll call it: "method"
    method = args.method

    # The third argument we take from command-line is 
    # the one that specify where the trained model's adapter file will be stored
    # we'll call it: "adapter_dir"
    adapter_dir = args.adapter_dir

    # The forth argument we take from command-line is 
    # the one that specify where the trained model's checkpoint file will be stored
    # we'll call it: "checkpoint_dir"
    checkpoint_dir = args.checkpoint_dir

    # we need to get the model's name,
    # we won't load the model now, because it could make the program crash
    MODEL_NAME = configs["model"]["model_name"]

    # print out some message for easy training process tracking 
    print(f" LOADING CONFIGURATION FOR {MODEL_NAME} FINE-TUNING COMPLETED ".center(terminal_width, "="))

    # we'll have to get the original dataset first
    DATASET_NAME = configs["model"]["dataset_name"]
    dataset = load_dataset(DATASET_NAME)

    # we need a tokenizer to tokenizing natural language
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Now we have original dataset and a tokenizer, 
    # the next step will be transforming original dataset into vector 
    # so that the model can perform calculation
    tokenized_dataset = load_tokenized_dataset(tokenizer, dataset, configs)

    # print out a message to track progress
    print(" PREPROCESSING DATA COMPLETED ".center(terminal_width, "="))

    # ----------------------------------- CHECKPOINT -----------------------------------
    
    # Now we load the model
    model = load_model(configs=configs, method=method)

    model.print_trainable_parameters()
    print(f" STEP 3: SETTING UP {MODEL_NAME} QUANTIZATION COMPLETED ".center(terminal_width, "="))

    # Load trainer
    trainer = load_trainer(
        configs=configs,
        tokenizer=tokenizer,
        tokenized_dataset=tokenized_dataset,
        model=model,
        checkpoint_dir=checkpoint_dir
    )

    # Start Training
    print(f" START TRAINING {MODEL_NAME}".center(terminal_width, "="))

    trainer.train()

    print(f" TRAINING {MODEL_NAME} COMPLETED ".center(terminal_width, "="))

    # Save Model adapter & checkpoint
    save_model(
        trainer=trainer,
        tokenizer=tokenizer,
        adapter_dir=adapter_dir,
        checkpoint_dir=checkpoint_dir
    )

if __name__ == "__main__":
    main()