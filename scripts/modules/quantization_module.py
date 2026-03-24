import torch
from transformers import (
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM
    )
from peft import prepare_model_for_kbit_training

def model_quantization(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto'
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)

    return base_model