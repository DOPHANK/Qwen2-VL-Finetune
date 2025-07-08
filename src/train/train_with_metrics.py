import os
import json
import logging
from PIL import Image
import torch
from transformers import HfArgumentParser, set_seed
from src.data.sft_dataset import make_supervised_data_module, process_vision_info
from src.trainer.sft_trainer import QwenSFTTrainer
from src.utils.params import DataArguments, ModelArguments, TrainingArguments
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from src.reward.reward_funcs import reward_accuracy, infos_accuracy

logger = logging.getLogger(__name__)


def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    return {
        "reward_accuracy": reward_accuracy(predictions, labels),
        "infos_accuracy": infos_accuracy(predictions, labels),
        "num_samples": len(predictions),
    }


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    set_seed(training_args.seed)

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32,
        device_map="auto"
    )

    model.config.use_cache = False
    data_module = make_supervised_data_module(tokenizer=processor.tokenizer, data_args=data_args)

    trainer = QwenSFTTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        **data_module
    )

    trainer.train(resume_from_checkpoint=False)

    # 🔍 Evaluate using same structure as training data
    if training_args.do_eval:
        eval_dataset = data_module.get("eval_dataset")
        if eval_dataset is not None:
            print("\n🔍 Running validation set evaluation with full structure...")
            predictions_output = trainer.predict(eval_dataset)
            preds = predictions_output.predictions
            labels = predictions_output.label_ids

            # 🖨️ Print comparison for first few examples
            decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            for i in range(min(5, len(decoded_preds))):
                print(f"\n📝 Example {i+1}:")
                print("🧠 Prediction:", decoded_preds[i])
                print("🎯 Ground Truth:", decoded_labels[i])

            print("\n📊 Validation Metrics:")
            for key, val in predictions_output.metrics.items():
                print(f"{key}: {val:.4f}")

    # Optional: single-image inference at the end
    if getattr(data_args, "inference_image_path", None):
        print("\n🖼️ Running test inference on single image...")

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_args.inference_image_path},
                        {"type": "text", "text": "Extract and list the filled information from this form as KEY: VALUE pairs."},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True
            ).to(training_args.device)

            model.eval()
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )

            output_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print("\n🧠🧾 Generated Output:")
            print(output_text)

        except Exception as e:
            print(f"[ERROR] Failed during single image inference: {e}")


if __name__ == "__main__":
    train()
