import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ViT5 using QDoRa for Text Summarization")
    
    parser.add_argument(
        "--configs", 
        type=str,
        required=True, # Bắt buộc phải truyền file config khi chạy
        help="Enter YAML training configs file path"
    )

    parser.add_argument(
        "--use_dora",
        type=bool,
        required=True,
        help="Which methode do you want to use? DORA --> True / LORA --> False"
    )

    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=False,
        default="model_adapter",
        help="Where do you want to save the model's adapter?"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
        default="model_checkpoint",
        help="Where do you want to save the model's checkpoint?"
    )
    
    return parser.parse_args()