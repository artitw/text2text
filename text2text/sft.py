import os
from dataclasses import dataclass, field

import torch
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig) 
from trl import SFTTrainer
from peft import LoraConfig

from datasets import DatasetDict, load_dataset

@dataclass
class DataTrainingArguments:
    """
    Defines data arguments and stores t2t default values.
    """
    dataset_name: str = field(
        default="smangrul/ultrachat-10k-chatml",
        metadata={"help": "Dataset for fine-tuning."})
    text_field: str = field(
        default="text",
        metadata={"help": "Field in dataset to use as input text."})
    splits: str = field(default="train,test")
    max_seq_length: int = field(default=2048)

@dataclass
class PeftModelArguments:
    """
    Defines and stores t2t default values for PEFT model configs.
    """
    peft_method: str = field(
        default="qlora",
        metadata={"help": "PEFT method to use. Only support QLoRA currently."})
    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha parameter for Lora scaling."})
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout probability for Lora layers."})
    lora_r: float = field(
        default=8,
        metadata={"help": "Lora attention dimension (Rank)."})
    target_modules: str = field(
        default="all-linear",
        metadata={"help": ("Modules to apply to adapter to."
                           "Alternatively: 'q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj'"
                           )
                        }
                    )
    use_reentrant: bool = field(
        default=True,
        metadata={"help": "For gradient checkpointing."})


@dataclass
class PeftTrainingArguments(TrainingArguments):
    """
    Stores t2t default values for training args.
    """
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs to train for."})
    eval_strategy: str = field(
        default="epoch",
        metadata={"help": "Evaluation strategy to use."})
    logging_steps: int = field(
        default=5,
        metadata={"help": "Number of steps to log at."})
    push_to_hub: bool = field(
        default=True,
        metadata={"help": "Uplaod model to Huggingface hub or not."})
    hub_private_repo: bool = field(default=True)
    # bf16: bool = field(default=True)
    learning_rate: float = field(default=1e-4)
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "LR scheduler type to use."})
    weight_decay: str = field(
        default=1e-4,
        metadata={"help": "Weight decay for AdamW optimizer."})
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Default value to support memory constraints."})
    gradient_checkpointing: bool = field(default=True)


class SFT:
    def __init__(self, model_name, output_dir, **kwargs):
        self.llm = model_name
        self.model_args = PeftModelArguments()
        self.data_args = DataTrainingArguments()
        if not output_dir:
            output_dir = f"{model_name.split("/")[-1]}-t2t-sft"
        self.train_args = PeftTrainingArguments(
            output_dir=output_dir)

        # Update arguments in parsed
        for kwarg in kwargs:
            if hasattr(self.model_args, kwarg):
                setattr(self.model_args, kwarg, kwargs[kwarg])
            elif hasattr(self.data_args, kwarg):
                setattr(self.data_args, kwarg, kwargs[kwarg])
            elif hasattr(self.train_args, kwarg):
                setattr(self.train_args, kwarg, kwargs[kwarg])
            else:
                raise AttributeError("Invalid Argument.")
        # Todo: option to use HF hub or not
            
    def prepare_model(self):
        # if self.model_args.peft_method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.uint8
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.llm,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float32
        )

        peft_config = LoraConfig(
            lora_alpha= self.model_args.lora_alpha,
            lora_dropout= self.model_args.lora_dropout,
            r=self.model_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.model_args.target_modules
        )
        
        tokenizer = AutoTokenizer(self.llm, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer, peft_config

    def prepare_dataset(self):
        raw_datasets = DatasetDict()
        for split in self.data_args.splits:
            try:
                dataset = load_dataset(self.data_args.dataset_name, 
                                       split=split)
                raw_datasets[split] = dataset
            except:
                print(f"Split type {split} not recognized as part of {self.dataset_name}")
                pass

        train_split = raw_datasets["train"]
        test_split = raw_datasets["test"]
        print(f"Size of training split: {len(train_split)}, Size of test split: {len(test_split)}")

        return train_split, test_split

    def train(self):
        model, tokenizer, peft_config = self.prepare_model()
        train_dataset, test_dataset = self.prepare_dataset()

        # Gradient checkpointing
        model.config.use_cache = not self.train_args.gradient_checkpointing
        if self.train_args.gradient_checkpointing:
            self.train_args.gradient_checkpointing_kwargs = {
                "use_reentrant": self.model_args.use_reentrant
            }

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            args=self.train_args,
            peft_config=peft_config,
            packing=True,
            dataset_kwargs={
                "append_concat_token": False,
                "add_special_tokens": False
            },
            dataset_text_field=self.data_args.text_field,
            max_seq_length=self.data_args.max_seq_length)

        trainer.accelerator.print(f"{trainer.model}")
        if hasattr(trainer.model, "print_trainable_parameters"):
            trainer.model.print_trainable_parameters()

        # Train
        checkpoint = None
        if self.train_args.resume_from_checkpoint is not None:
            checkpoint = self.train_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)

        # Save model
        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.save_model()

#%env XLA_USE_BF16=1