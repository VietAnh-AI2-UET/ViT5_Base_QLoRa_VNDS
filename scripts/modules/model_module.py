from .quantization_module import get_quantization_model
from .lora_module import get_peft_configs
from peft import (
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)
from ..utils.training_utils import (
    get_peft_configs_kwargs
)

def get_model_for_training(configs, method):
    """
    Load the pre-trained model by quantization technique
    and combine it with specified PEFT method 
    """
    # Load pre-trained model with quantization setting
    MODEL_NAME=configs["model"]["model_name"]
    quantization_base_model = get_quantization_model(
        base_model=MODEL_NAME
    )

    # Set this pre-trained model ready for fine-tuning phase
    quantization_base_model = prepare_model_for_kbit_training(
        quantization_base_model, 
        use_gradient_checkpointing=False
    )

    # Get LORA DORA ADALORA BLABLA hyperparameters dictionary / kwargs
    peft_configs_kwargs = get_peft_configs_kwargs(
        configs=configs,
        method=method
    )

    # Feed the peft_configs_kwargs that was made into the function of the real peft library
    peft_configs = get_peft_configs(
        peft_configs_kwargs=peft_configs_kwargs,
        method=method,
    )

    # Combine the base pre-trained model with the peft_configs
    model = get_peft_model(
        quantization_base_model,
        peft_configs
    )

    return model

def get_fine_tuned_model(configs, adapter_dir):
    """
    Load the pre-trained model by quantization technique
    And combine it with the fine_tuned adapter
    """
    MODEL_NAME=configs["model"]["model_name"]
    quantization_base_model = get_quantization_model(
        base_model=MODEL_NAME
    )

    fine_tuned_model = PeftModel.from_pretrained(
        quantization_base_model,
        adapter_dir
    )

    return fine_tuned_model
