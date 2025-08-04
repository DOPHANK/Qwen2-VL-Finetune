import os
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2_5_VLForConditionalGeneration
from src.trainer import QwenSFTTrainer
from src.dataset import make_supervised_data_module
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl
from src.train.monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward
from torch.nn import CrossEntropyLoss

import deepspeed
from src.train.reward_funcs import accuracy_reward, format_reward, accuracy_infos
import numpy as np
from PIL import Image

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    #if verbose:
    #    rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(params, requires_grad=True):
    for p in params:
        if p.dtype.is_floating_point or p.is_complex():
            p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

from transformers.trainer_utils import EvalPrediction





def compute_metrics(eval_preds):
    try:
        predictions, labels = eval_preds
    except Exception as e:
        raise RuntimeError(f"❌ Failed to unpack eval_preds: {e}, got {type(eval_preds)} with content: {repr(eval_preds)}")

    try:
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        if isinstance(predictions, np.ndarray):
            if predictions.dtype != np.int32 and predictions.dtype != np.int64:
                predictions = np.rint(predictions).astype(np.int32)
            predictions = predictions.tolist()

        if isinstance(labels, np.ndarray):
            labels = labels.tolist()

        # ✅ Replace -100 (ignore index) with pad_token_id
        labels = [
            [token if token != -100 else tokenizer.pad_token_id for token in label]
            for label in labels
        ]
    except Exception as e:
        raise RuntimeError(f"❌ Failed preprocessing predictions/labels: {e}")

    try:
        # Basic value checks to avoid large garbage
        for seq in predictions:
            for token in seq:
                if token < 0 or token > tokenizer.vocab_size * 2:  # allow some room for added tokens
                    raise ValueError(f"⚠️ Invalid token ID in prediction: {token}")

        for seq in labels:
            for token in seq:
                if token < 0 or token > tokenizer.vocab_size * 2:
                    raise ValueError(f"⚠️ Invalid token ID in label: {token}")

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        #for i in range(min(5, len(decoded_preds))):
        #            print(f"\n📝 Example {i+1}:")
        #            print("🧠 Prediction:", decoded_preds[i])
        #            print("🎯 Ground Truth:", decoded_labels[i])
    except Exception as e:
        raise RuntimeError(f"❌ Failed to decode predictions/labels: {e}")

    # Wrap decoded text as chat-style input
    completions = [[{"role": "assistant", "content": pred.strip()}] for pred in decoded_preds]
    references = [{"role": "assistant", "content": ref.strip()} for ref in decoded_labels]

    rewards_hw = accuracy_infos(completions, references)
    rewards_all = accuracy_reward(completions, references)
    
    mean_reward_hw = sum(rewards_hw) / len(rewards_hw) if rewards_hw else 0.0
    mean_reward_all = sum(rewards_all) / len(rewards_all) if rewards_all else 0.0

    return {
        "reward_accuracy": mean_reward_all,
        "infos_accuracy": mean_reward_hw,
        "num_samples": len(rewards_hw),
    }




def train():
    global local_rank, tokenizer

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.environ["OUTPUT_DIR"] = training_args.output_dir
    os.environ["DEBUG_MODE"] = str(training_args.debug_mode_activate)
    
    use_liger = training_args.use_liger
    if "Qwen2.5" in model_args.model_id:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen2_5_with_mixed_modality_forward(use_liger=use_liger)
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
    else:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen_2_with_mixed_modality_forward(use_liger=use_liger)
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=False)
    

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            low_cpu_mem_usage = True,
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    # Add-ins
    if training_args.bits not in [4, 8] and training_args.deepspeed==None:
        bnb_model_from_pretrained_args["device_map"] = {
            "": 0,  # let it auto-balance
        }

    if "Qwen2.5" in model_args.model_id:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            **bnb_model_from_pretrained_args
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        model = get_peft_model(model, peft_config)

        # Peft maodel makes vision tower and merger freezed again.
        # Configuring fuction could be called here, but sometimes it does not work properly.
        # So I just made it this way.
        # Need to be fixed in the future.

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True
    
    processor = AutoProcessor.from_pretrained(model_args.model_id, use_fast=True)
    tokenizer = processor.tokenizer

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)

    # Handle AMP crash due to frozen FP16
    if not training_args.freeze_vision_tower:
        # Fix for AMP crashing due to frozen FP16 parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad and param.dtype == torch.float16:
                    param.data = param.data.cpu().float()  # ✅ move to CPU before casting
        
        # Debug log for remaining frozen fp16 parameters
        frozen_fp16 = [
            name for name, param in model.named_parameters()
            if not param.requires_grad and param.dtype == torch.float16
        ]
        
        # Optional: cast trainable params to float32
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.dtype != torch.float32:
                    param.data = param.data.float()
    
    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # === Run testing on test set ===
    if getattr(data_args, "test_data_path", None):
        try:
            test_dataset = data_module["test_dataset"]
        
            test_output = trainer.predict(test_dataset)
            
            test_predictions = test_output.predictions
            
            test_labels = test_output.label_ids
            
            test_metrics = test_output.metrics
        except Exception as e:
            rank0_print(f"[ERROR] Failed during evaluation: {e}")
        
    # === Custom single image generation test ===
    import time
    from pathlib import Path
    from qwen_vl_utils import process_vision_info
    
    # === CONFIGURATION ===
    batch_size = 2                                    # Process N images at a time
    max_new_tokens = 256                              # Generation length
    max_dim = 1024                                    # Resize max dimension
    
    # === Logging Helper ===
    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    
    # === Collect Images from Directory ===
    if getattr(data_args, "inference_image_path", None):
        inference_image_dir = data_args.inference_image_path
        
        image_paths = sorted([
            str(p) for p in Path(inference_image_dir).glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
        log(f"Found {len(image_paths)} images in {inference_image_dir}")
        
        # === GPU Info ===
        if torch.cuda.is_available():
            log(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            log("⚠️ No GPU detected! Running on CPU (very slow)")
        
        # === Prepare Messages Template ===
        def build_message_for_image(img):
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": (
                            "Now extract infos from THIS image as KEY: VALUE pairs in ChatML format.\n"
                            "Follow this format strictly: <im_start>KEY: VALUE<im_end>.\n"
                            "Start example from <image>/kaggle/working/images/8/1\n"
                            "<im_start>Side Code: 1<im_end>\n"
                            "<im_start>ID: 8<im_end>\n"
                            "<im_start>RECORD_DTC: 20/11/2024<im_end>\n"
                            "<im_start>SEX: Male<im_end>\n"
                            "<im_start>AGE: 28.0<im_end>\n"
                            "<im_start>ADMISSION_DTC: 05/09/2019<im_end>\n"
                            "<im_start>DISCHARGE_DTC: 13/09/2019<im_end>\n"
                            "<im_start>ILLNESS_DAYS: 8.0<im_end>\n"
                            "<im_start>TEMP_ADM: 38.1<im_end>\n"
                            "<im_start>SYSBP: 117.0<im_end>\n"
                            "<im_start>DIABP: 62.0<im_end>\n"
                            "<im_start>HR: 142.0<im_end>\n"
                            "<im_start>RESP: 34.0<im_end>\n"
                            "<im_start>SPO2: 100.0<im_end>\n"
                            "<im_start>CONSCIOUS_LEVEL: Unconscious<im_end>\n"
                            "<im_start>WEIGHT: nan<im_end>\n"
                            "<im_start>NA_W: True<im_end>\n"
                            "<im_start>HEIGHT: nan<im_end>\n"
                            "<im_start>NA_H: True<im_end>\n"
                            "<im_start>HYPERTENSION: False<im_end>\n"
                            "<im_start>DIABETES: False<im_end>\n"
                            "<im_start>DYSLIPIDAEMIA: False<im_end>\n"
                            "<im_start>IHD: False<im_end>\n"
                            "<im_start>CLUNGD: False<im_end>\n"
                            "<im_start>CVD: False<im_end>\n"
                            "<im_start>CLIVERD: False<im_end>\n"
                            "<im_start>CKD: False<im_end>\n"
                            "<im_start>MALIGNANCY: False<im_end>\n"
                            "<im_start>AUTOIMMUNE_DISEASE: False<im_end>\n"
                            "<im_start>OTH_MORBIDITIES: nan<im_end>"
                            "End example\n"
                            "Remember: Checkboxes are indicated at the start of the VALUE in the image."
                        )}
                    ]
                }
            ]
        
        # === MAIN INFERENCE ===
        start_time = time.time()
        all_outputs = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            log(f"\n🔹 Processing batch {i//batch_size + 1} ({len(batch_paths)} images)")
        
            messages_batch = []
            images_loaded = []
        
            # === Load and Resize Images ===
            for img_path in batch_paths:
                t_load = time.time()
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                if max(w, h) > max_dim:
                    scale = max_dim / max(w, h)
                    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                log(f"🖼️ Loaded {Path(img_path).name} size={img.size} in {time.time()-t_load:.2f}s")
                images_loaded.append(img)
                messages_batch.append(build_message_for_image(img))
        
            # === Build Text Prompts ===
            text_batch = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages_batch
            ]
        
            # === Preprocess ===
            t0 = time.time()
            image_inputs, video_inputs = process_vision_info(messages_batch)
            inputs = processor(
                text=text_batch,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            log(f"✅ Preprocessed batch in {time.time() - t0:.2f}s")
        
            # === Log GPU Memory Before Generation ===
            if torch.cuda.is_available():
                log(f"💾 GPU Memory before generate: {torch.cuda.memory_allocated()/1e6:.1f} MB")
        
            # === Inference ===
            t0 = time.time()
            model.eval()
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
            log(f"🚀 Generation took {time.time() - t0:.2f}s")
        
            # === Log GPU Memory After Generation ===
            if torch.cuda.is_available():
                log(f"💾 GPU Memory after generate: {torch.cuda.memory_allocated()/1e6:.1f} MB")
        
            # === Decode Outputs ===
            t0 = time.time()
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            outputs = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            log(f"📝 Decoding took {time.time() - t0:.2f}s")
        
            # === Collect & Print Results ===
            for img_path, text in zip(batch_paths, outputs):
                log(f"\n🖼️ [Result for {Path(img_path).name}]:\n{text}")
                all_outputs.append({"image": img_path, "result": text})
        
        log(f"\n✅ Finished multi-image inference in {time.time() - start_time:.2f}s for {len(image_paths)} images.")

    model.config.use_cache = True

    #Add-ins
    model.to(device=training_args.device, dtype=compute_dtype)

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
