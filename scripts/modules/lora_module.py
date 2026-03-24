from peft import ( 
    LoraConfig,
    AdaLoraConfig
)

def get_peft_configs(peft_configs_kwargs, method):
    if method == "LORA":
        peft_configs = LoraConfig(**peft_configs_kwargs)
    elif method == "DORA":
        peft_configs = LoraConfig(**peft_configs_kwargs)
    elif method == "ADALORA":
        peft_configs = AdaLoraConfig(**peft_configs_kwargs)

    return peft_configs