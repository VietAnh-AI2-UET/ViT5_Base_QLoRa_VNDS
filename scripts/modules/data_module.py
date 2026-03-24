from datasets import load_dataset

def get_tokenized_dataset(configs, tokenizer):
    """
    This function help tokenizing the entire dataset
    """
    DATASET_NAME = configs["model"]["dataset_name"]
    TRAIN_SAMPLES = configs["data"]["train_samples"]
    VAL_SAMPLES = configs["data"]["val_samples"]

    # Load original dataset
    dataset = load_dataset(DATASET_NAME)

    # Subsample dataset
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))                   # We only need about 6000 samples for fine-tuning
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(VAL_SAMPLES))           # And only 1000 samples for validation

    # Define function for tokenizing string
    def preprocess_function(batch):
        """
        Batch (str) --> Batch (tensor)
        """
        input_texts = ['tóm tắt: ' + txt for txt in batch['article']]
        input_max_length = 1024
        target_max_length = 256

        model_inputs = tokenizer(input_texts, max_length=input_max_length, truncation=True)
        labels = tokenizer(text_target=batch['abstract'], max_length=target_max_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # Mass tokenizing dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset