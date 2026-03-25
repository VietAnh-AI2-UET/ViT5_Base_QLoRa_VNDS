import argparse

class BaseArgs:
    """Base class for parsing command-line arguments."""
    
    def __init__(self, description="Fine-tune ViT5 for Text Summarization"):
        self.parser = argparse.ArgumentParser(description=description)
        self.add_common_args()
    
    def add_common_args(self):
        """Add common arguments shared across training scripts."""
        self.parser.add_argument(
            "--configs", 
            type=str,
            required=True,
            help="Enter YAML training configs file path"
        )
        self.parser.add_argument(
            "--adapter_dir",
            type=str,
            required=False,
            default="model_adapter",
            help="Enter model's adapter saving location"
        )
    
    def parse_args(self):
        """Parse and return arguments."""
        return self.parser.parse_args()