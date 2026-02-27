import torch
import argparse
import yaml
import evaluate
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Inferencing and computing ROUGE, BERTscores")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML configs directory"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 0. Read YAML Configs
    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Define model, dataset and adapter
    MODEL_NAME = config["model"]["model_name"]
    DATASET_NAME = config["model"]["dataset_name"]
    ADAPTER_PATH = config["model"]["adapter_path"]

    # Define generation hyperparams
    OUTPUT_MAX_LENGTH = config["generation"]["output_max_length"]
    OUTPUT_MIN_LENGTH = config["generation"]["output_min_length"]
    NUM_BEAMS = config["generation"]["num_beams"]
    NO_REPEAT_NGRAM_SIZE = config["generation"]["no_repeat_ngram_size"]
    REPETITION_PENALTY = config["generation"]["repetition_penalty"]
    LENGTH_PENALTY = config["generation"]["length_penalty"]

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    # 2. Load Dataset
    dataset = load_dataset(DATASET_NAME)
    test_dataset = dataset["test"].shuffle(seed=42).select(range(50))       # This dataset is for quick evaluation

    # 3. Load Quantization configs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # 4. Load 4-bits base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map='auto'
    )

    # 5. Combine base model and adapter
    qlora_vit5_baseline = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # 6. Create output generating function
    def generate_summary(tokenizer, model, text) -> str:
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
    
    # 7. Start generating summarization
    test_articles = [item["article"] for item in test_dataset]
    reference_summaries = [item["abstract"] for item in test_dataset]

    generated_summaries = []
    for article in tqdm(test_articles, desc="Generating summaries"): # Use tqdm for progress bar
        generated_summaries.append(generate_summary(tokenizer, qlora_vit5_baseline, article))

    # 8. Calculate ROUGE and BERT_scores
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