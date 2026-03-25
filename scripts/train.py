import shutil
import yaml
from datasets import load_dataset
from scripts.modules.arguments import BaseArgs
from scripts.modules.data_module import get_tokenized_dataset
from scripts.modules.model_module import get_model_for_training
from scripts.utils.save_model_utils import save_model
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from scripts.utils.training_utils import (
    get_training_args_kwargs
)

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
    
def main():
    # Print out running scripts
    terminal_width = shutil.get_terminal_size().columns
    print(" RUNNING TRAIN.PY ".center(terminal_width, "="))

    # Load command-line arguments into this script
    args = TrainArgs().parse_args()

    # Load configs from YAML configuration file
    with open(args.configs, 'r', encoding='utf-8') as file:
        configs = yaml.safe_load(file)

    # ------------------------------------------------------------ TRAINING CONSTANTS ------------------------------------------------------------   
    MODEL_NAME = configs["model"]["model_name"]
    DATASET_NAME = configs["model"]["dataset_name"]
    METHOD = args.method
    ADAPTER_DIR = args.adapter_dir
    CHECKPOINT_DIR = args.checkpoint_dir

    print(f" LOADING CONFIGURATION FOR {MODEL_NAME} FINE-TUNING COMPLETED ".center(terminal_width, "="))

    # ------------------------------------------------------------ PREPROCESSING DATA ------------------------------------------------------------
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
    
    print(" PREPROCESSING DATA COMPLETED ".center(terminal_width, "="))

    # ------------------------------------------------------------ SETUP QUANTIZATION LORA MODEL FOR TRAINING ------------------------------------------------------------

    # Initiate model for fine-tuning
    model = get_model_for_training(
        configs=configs,
        method=METHOD
    )

    model.print_trainable_parameters()

    print(f" STEP 3: SETTING UP {MODEL_NAME} QUANTIZATION COMPLETED ".center(terminal_width, "="))

    # ------------------------------------------------------------ SETUP HYPERPARAMETERS & TRAINER ------------------------------------------------------------

    # Get training arguments kwargs
    training_args_kwargs = get_training_args_kwargs(
        configs=configs, 
        checkpoint_dir=CHECKPOINT_DIR
    )

    # Feed the training arguments kwargs into Seq2Seq library
    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    # Use datacollator for padding 
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model
    )

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

    # Start Training
    print(f" START TRAINING {MODEL_NAME}".center(terminal_width, "="))

    trainer.train()

    # Save Model adapter & checkpoint
    save_model(
        trainer=trainer,
        tokenizer=tokenizer,
        adapter_dir=ADAPTER_DIR,
        checkpoint_dir=CHECKPOINT_DIR
    )

if __name__ == "__main__":
    main()