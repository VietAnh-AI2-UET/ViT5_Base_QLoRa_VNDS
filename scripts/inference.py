import shutil
import torch
import yaml
import evaluate
import logging
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from .modules.arguments import BaseArgs
from .modules.model_module import get_fine_tuned_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferArgs(BaseArgs):
    def __init__(self, description="Infering ViT5 for Text Summarization"):
        super().__init__(description)
        self.add_adapter_output()

    def add_adapter_output(self):
        self.parser.add_argument(
            "--adapter_dir",
            type=str,
            required=False,
            default="model_adapter",
            help="Enter model's adapter saving location"
        )
        
def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_data(configs: dict, adapter_dir: str) -> tuple:
    """Load tokenizer and test dataset."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    dataset = load_dataset(configs["model"]["dataset_name"])
    test_dataset = dataset["test"].shuffle(seed=42).select(range(50))
    return tokenizer, test_dataset

def load_model(configs: dict, adapter_dir: str):
    """Load fine-tuned model."""
    return get_fine_tuned_model(configs=configs, adapter_dir=adapter_dir)

def generate_summaries(configs: dict, tokenizer, model, test_articles: list) -> list:
    """Generate summaries for test articles."""
    generated_summaries = []
    for article in tqdm(test_articles, desc="Generating summaries"):
        summary = generate_summary(configs, tokenizer, model, article)
        generated_summaries.append(summary)
    return generated_summaries

def generate_summary(configs: dict, tokenizer, model, text: str) -> str:
    """Generate summary for a single text."""
    OUTPUT_MAX_LENGTH = configs["generation"]["output_max_length"]
    OUTPUT_MIN_LENGTH = configs["generation"]["output_min_length"]
    NUM_BEAMS = configs["generation"]["num_beams"]
    NO_REPEAT_NGRAM_SIZE = configs["generation"]["no_repeat_ngram_size"]
    REPETITION_PENALTY = configs["generation"]["repetition_penalty"]
    LENGTH_PENALTY = configs["generation"]["length_penalty"]

    inputs = tokenizer("tóm tắt: " + text, max_length=1024, truncation=True, return_tensors='pt').to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs['input_ids'],
            max_length=OUTPUT_MAX_LENGTH,
            min_length=OUTPUT_MIN_LENGTH,            
            num_beams=NUM_BEAMS,              
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,   
            repetition_penalty=REPETITION_PENALTY,   
            early_stopping=True,
            length_penalty=LENGTH_PENALTY,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def compute_metrics(generated_summaries: list, reference_summaries: list):
    """Compute ROUGE and BERTScore metrics."""
    # ROUGE
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(
        predictions=generated_summaries,
        references=reference_summaries,
        use_stemmer=True
    )
    logger.info("ROUGE Scores:")
    for key, value in rouge_results.items():
        logger.info(f"{key}: {value:.4f}")

    # BERTScore
    logger.info("\nCalculating BERTScore (this may take a while)...")
    bertscore = evaluate.load('bertscore')
    bertscore_results = bertscore.compute(
        predictions=generated_summaries,
        references=reference_summaries,
        model_type='bert-base-multilingual-cased',
        lang='vi'
    )
    logger.info("\nBERTScore Results:")
    logger.info(f"BERTScore F1 (mean): {sum(bertscore_results['f1']) / len(bertscore_results['f1']):.4f}")
    logger.info(f"BERTScore Precision (mean): {sum(bertscore_results['precision']) / len(bertscore_results['precision']):.4f}")
    logger.info(f"BERTScore Recall (mean): {sum(bertscore_results['recall']) / len(bertscore_results['recall']):.4f}")

def main():
    try:
        # Print out running scripts
        terminal_width = shutil.get_terminal_size().columns
        print(" RUNNING INFERENCE.PY ".center(terminal_width, "="))
        
        # Load command-line arguments
        args = InferArgs().parse_args()

        # Load config
        configs = load_config(args.config)

        # Load data
        tokenizer, test_dataset = load_data(configs, args.adapter_dir)

        # Load model
        model = load_model(configs, args.adapter_dir)

        # Extract data
        test_articles = [item["article"] for item in test_dataset]
        reference_summaries = [item["abstract"] for item in test_dataset]

        # Generate summaries
        generated_summaries = generate_summaries(configs, tokenizer, model, test_articles)

        # Compute metrics
        compute_metrics(generated_summaries, reference_summaries)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()