import os
import torch
import torch.nn as nn
from typing import Optional, List
from datasets import Dataset

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)
from src.train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3

from transformers.trainer_utils import PredictionOutput

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class QwenSFTTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(QwenSFTTrainer, self).__init__(*args, **kwargs)

    def predict(
        self,
        test_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
    ) -> PredictionOutput:

        print("\nPredicting...")

        self.model.eval()
        self._memory_tracker.start()
    
        test_dataloader = self.get_test_dataloader(test_dataset)

        output = self.prediction_loop(
            test_dataloader,
            description="Prediction",
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
    
        # 🔁 If model is generative, use generate()
        all_preds = []
        all_labels = []
    
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(self.args.device)
            attention_mask = batch["attention_mask"].to(self.args.device)
    
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    do_sample=False
                )
    
            all_preds.extend(generated_ids.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
    
        self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

        # ✅ Now you have predictions = token IDs
        metrics = {}
        if self.compute_metrics is not None:
            metrics = self.compute_metrics((all_preds, all_labels))

        self._memory_tracker.stop_and_update_metrics(metrics)
    
        return PredictionOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics)

        def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            predict_with_generate: bool = True,
        ):
            dataset = self.eval_dataset if eval_dataset is None else eval_dataset
            if dataset is None or len(dataset) == 0:
                print("⚠️ No evaluation dataset provided. Skipping evaluation.")
                return {}
    
            # ✅ Prepare log file
            log_file = os.path.join(self.args.output_dir, "test_predictions.log")
            os.makedirs(self.args.output_dir, exist_ok=True)
            print(f"🧪 Starting evaluation on {len(dataset)} samples...")
            f = open(log_file, "w")
    
            if predict_with_generate:
                print("\nEvaluating with generate...")
                self.model.eval()
            
                dataloader = self.get_eval_dataloader(dataset)
                for step, inputs in enumerate(dataloader):
                    img_path = inputs.get("image_path", ["N/A"])
                    if isinstance(img_path, list):
                        img_path = img_path[0]
    
                    print(f"\n--- [Step {step}] Image: {img_path} ---")
                    f.write(f"Step {step} | Image: {img_path}\n")
    
                    # ✅ Run prediction
                    input_ids = inputs["input_ids"].to(self.args.device)
                    attn_mask = inputs["attention_mask"].to(self.args.device)
                    labels = inputs.get("labels", None)
    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            max_new_tokens=256,
                            do_sample=False
                        )
    
                    # ✅ Decode outputs
                    pred_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
                    label_text = "[NO LABEL]"
                    if labels is not None:
                        label_text = self.tokenizer.decode(labels[0], skip_special_tokens=False)
    
                    # ✅ Log to console & file
                    print(f"✅ Ground Truth: {label_text}")
                    print(f"🤖 Model Output: {pred_text}")
                    f.write(f"Ground Truth: {label_text}\nPrediction: {pred_text}\n{'-'*50}\n")
    
                f.close()
    
                # ✅ Call original predict to compute metrics
                output = self.predict(dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
                if self.compute_metrics is not None:
                    metrics = self.compute_metrics((output.predictions, output.label_ids))
                    output.metrics.update(metrics)
                    print("📊 Metrics:", metrics)
                print(f"📝 Detailed predictions saved to {log_file}")
                return output
            else:
                print("\nEvaluating without generate...")
                return super().evaluate(
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix
                )

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer
    
    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
                best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
                best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

                if os.path.exists(best_checkpoint_dir):
                    self.state.best_model_checkpoint = best_checkpoint_dir

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                self._save_scaler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
                for cb in [
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)
        else:
            super(QwenSFTTrainer, self)._save_checkpoint(model, trial)

    # def training_step(self, model, inputs):
    #     for name, param in model.named_parameters():
    #         if 'visual' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
    # 
    #     return super().training_step(model, inputs)
