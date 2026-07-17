

def get_tokenized_dataset(configs, dataset, tokenizer):
    """
    This function help tokenizing the entire dataset
    """

    # first, we need to specify how many train and validation sample will be use
    # if you want to adjust the number of sample, go to the YAML config file
    TRAIN_SAMPLES = configs["data"]["train_samples"]
    VAL_SAMPLES = configs["data"]["val_samples"]

    # Subsample the original dataset
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))                   # We only need about 6000 samples for fine-tuning
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(VAL_SAMPLES))           # And only 1000 samples for validation

    # Define function for tokenizing string
    # this inner function take in a batch of sample so we'll have to process it like a batch
    def preprocess_function(batch):
        """
        Batch (str) --> Batch (tensor)
        """

        # the model we're using is ViT5-Base, which is a T5 model, 
        # so that we have to add a task specific prefix before every sample 
        # for the model to understand what to do
        input_texts = ['tóm tắt: ' + txt for txt in batch['article']]

        # after some analysis, I found that the platform we're using to train model (Kaggle)
        # only work well if the input length be 1000 and label max length be 240
        input_max_length = 1000
        target_max_length = 240

        # Perform tokenized while truncating the input of original data
        model_inputs = tokenizer(input_texts, max_length=input_max_length, truncation=True)
        # Perform tokenized while truncating the label of original data
        labels = tokenizer(text_target=batch['abstract'], max_length=target_max_length, truncation=True)
        
        # After that, we'll have to add the "input_ids" field,
        # which is the tokenized version of original label
        # into "model_inputs" variable and let that new field be: "lables"
        model_inputs['labels'] = labels['input_ids']

        # we return the variable "model_inputs": a dictionary contain: 
        # "model_input["input_ids"]": the tokenized version of input paragraph
        # "model_input["labels"]": the tokenized version of label paragraph
        return model_inputs
    
    # Mass tokenizing dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset