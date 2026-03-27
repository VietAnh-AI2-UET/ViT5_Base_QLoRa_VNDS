import os
import shutil

def save_model(trainer, tokenizer, adapter_dir, checkpoint_dir):
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    shutil.make_archive(adapter_dir, "zip", adapter_dir)
    print(f"Thư mục '{adapter_dir}' đã được nén thành '{adapter_dir}.zip'")

    shutil.make_archive(checkpoint_dir, "zip", checkpoint_dir)
    print(f"Thư mục '{checkpoint_dir}' đã được nén thành '{checkpoint_dir}.zip'")