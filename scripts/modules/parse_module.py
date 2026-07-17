import argparse

class BaseArgs:
    """Base class for parsing command-line arguments."""
    
    def __init__(self, description="Fine-tune ViT5 for Text Summarization"):
        # init the object named "parser" to read arguments passed to the script
        self.parser = argparse.ArgumentParser(description=description)
        self.add_common_args()
    
    def add_common_args(self):
        """Add common arguments shared across training scripts."""
        # use the methode ".add_argument" of ArgumentParser to initiate an argument
        # let this argument name be "config", and "config" will be the path to the YAML config file
        self.parser.add_argument(
            "--config", 
            type=str,
            required=True,
            help="Enter YAML training configs file path"
        )
    
    def parse(self):
        """Parse and return arguments."""
        # the function named ".parse_args()" is used to force the object "parser" 
        # to read these arguments we pass into terminal
        return self.parser.parse_args()