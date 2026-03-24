from peft import (
    get_peft_model, 
    LoraConfig
    )

def lora_configuration(base_model, lora_r, lora_alpha, lora_target_module, lora_dropout, use_dora):
    lora_configs_kwargs = {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_target_module,
        "target_modules": lora_dropout,
        "bias": "none",
        "task_type": "SEQ_2_SEQ_LM",
        "use_dora": use_dora
    }
    
    lora_configs = LoraConfig(**lora_configs_kwargs)
    model = get_peft_model(base_model, lora_configs)

    return model