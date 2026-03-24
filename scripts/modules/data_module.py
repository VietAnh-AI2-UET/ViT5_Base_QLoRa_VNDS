from datasets import load_dataset

def prepare_dataset(dataset_name, tokenizer, train_samples, val_samples):
    dataset = load_dataset(dataset_name)

    # Subsample dataset
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(train_samples))                   # We only need about 6000 samples for fine-tuning
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(val_samples))           # And only 1000 samples for validation

    def preprocess_function(batch):
        input_texts = ['tóm tắt: ' + txt for txt in batch['article']]
        input_max_length = 1024
        target_max_length = 256

        model_inputs = tokenizer(input_texts, max_length=input_max_length, truncation=True)
        labels = tokenizer(text_target=batch['abstract'], max_length=target_max_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset