import shutil
import torch
import yaml
import evaluate
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from scripts.modules.arguments import BaseArgs
from scripts.modules.model_module import get_fine_tuned_model

class InferArgs(BaseArgs):
    def __init__(self, description="Infering ViT5 for Text Summarization"):
        super().__init__(description)

def generate_summary(configs, tokenizer, model, text) -> str:
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

    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

def main():
    # Print out running scripts
    terminal_width = shutil.get_terminal_size().columns
    print(" RUNNING TRAIN.PY ".center(terminal_width, "="))
    
    # Load command-line arguments
    args = InferArgs().parse_args()

    # Load YAML configs
    with open(args.config, 'r', encoding='utf-8') as file:
        configs = yaml.safe_load(file)

    # ------------------------------------------------------------ TRAINING CONSTANTS ------------------------------------------------------------   
    DATASET_NAME = configs["model"]["dataset_name"]
    ADAPTER_DIR = args.adapter_dir

    # ------------------------------------------------------------ LOAD DATASET ------------------------------------------------------------   
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    # Load original dataset
    dataset = load_dataset(DATASET_NAME)

    # Select only test set from original dataset
    test_dataset = dataset["test"].shuffle(seed=42).select(range(50))

    # ------------------------------------------------------------ LOAD FINE-TUNED MODEL ------------------------------------------------------------   
    # Load fine tuned model
    fine_tuned_model = get_fine_tuned_model(
        configs=configs,
        adapter_dir=ADAPTER_DIR
    )

    # ------------------------------------------------------------ GENERATE SUMMARIES ------------------------------------------------------------       
    # Extract articles from test dataset
    test_articles = [item["article"] for item in test_dataset]

    # Extract summarization of test_articles from test dataset
    reference_summaries = [item["abstract"] for item in test_dataset]

    generated_summaries = []
    for article in tqdm(test_articles, desc="Generating summaries"): # Use tqdm for progress bar
        generated_summaries.append(generate_summary(configs, tokenizer, fine_tuned_model, article))

    # ------------------------------------------------------------ CALCULATE ROUGE AND BERTSCORE ------------------------------------------------------------   
    # Initiate ROUGE and BERT_scores computing object
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')

    # Compute ROUGE scores
    rouge_results = rouge.compute(
        predictions=generated_summaries,
        references=reference_summaries,
        use_stemmer=True  # Use stemming for more accurate ROUGE computation
    )

    print("ROUGE Scores:")
    for key, value in rouge_results.items():
        print(f"{key}: {value:.4f}")

    # Compute BERTScore
    # Note: BERTScore can be slow. You might want to sample a smaller subset if it takes too long.
    # For Vietnamese, you might need a specific BERT model. 'bert-base-multilingual-cased' is a good general choice.
    # Or 'vinai/phobert-base' if you have it installed and it's compatible.

    print("\nCalculating BERTScore (this may take a while)...")
    bertscore_results = bertscore.compute(
        predictions=generated_summaries,
        references=reference_summaries,
        model_type='bert-base-multilingual-cased', # Using a multilingual BERT model
        lang='vi' # Specify language
    )

    print("\nBERTScore Results:")
    # Print average F1 score
    print(f"BERTScore F1 (mean): {sum(bertscore_results['f1']) / len(bertscore_results['f1']):.4f}")
    print(f"BERTScore Precision (mean): {sum(bertscore_results['precision']) / len(bertscore_results['precision']):.4f}")
    print(f"BERTScore Recall (mean): {sum(bertscore_results['recall']) / len(bertscore_results['recall']):.4f}")

if __name__ == "__main__":
    main()