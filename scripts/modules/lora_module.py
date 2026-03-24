from peft import (
    get_peft_model, 
    LoraConfig,
    AdaLoraConfig
)

def lora_configuration(base_model, lora_configs_kwargs):
    lora_configs = LoraConfig(**lora_configs_kwargs)
    model = get_peft_model(base_model, lora_configs)

    return model

def adalora_configuration(base_model, adalora_configs_kwargs):
    adalora_configs = AdaLoraConfig(**adalora_configs_kwargs)
    model = get_peft_model(base_model, adalora_configs)

    return model