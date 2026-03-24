import torch
from transformers import (
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM
    )

def get_quantization_model(model_name):
    """
    Combine pre-trained model with quantization technique
    """
    # Setup quantization configs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load pre-trained model with quantization configs
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto'
    )

    return base_model