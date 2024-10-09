import os
import logging
from dataclasses import dataclass, field

import torch
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig) 
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

from datasets import DatasetDict, IterableDatasetDict, load_dataset
from huggingface_hub import login, notebook_login

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s')

@dataclass
class DataTrainingArguments:
    """
    Defines and stores t2t default values for data args.
    """
    dataset_name: str = field(
        default="smangrul/ultrachat-10k-chatml",
        metadata={"help": "Dataset for fine-tuning."})
    text_field: str = field(
        default="text",
        metadata={"help": "Field in dataset to use as input text."})
    splits: str = field(default="train,test")
    max_seq_length: int = field(default=2048)
    append_concat_token: bool = field(default=False)
    add_special_tokens: bool = field(default=False)

@dataclass
class PeftModelArguments:
    """
    Defines and stores t2t default values for PEFT model configs.
    """
    peft_method: str = field(
        default="qlora",
        metadata={"help": "PEFT method to use. Only supports QLoRA currently."})
    model_dtype: str = field(
        default="float16",
        metadata={"help": "Load model in this dtype." 
                  "fp16 for colab's T4, can be modified for other GPU machines."})
    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha parameter for LoRA scaling."})
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout probability for LoRA layers."})
    lora_r: float = field(
        default=8,
        metadata={"help": "LoRA attention dimension (Rank)."})
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
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Flash attention for training."})

@dataclass
class PeftTrainingArguments(SFTConfig):
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
        metadata={"help": "Upload model to Huggingface hub or not."})
    hub_private_repo: bool = field(default=True)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
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
        metadata={"help": "Set here to support memory constraints."})
    gradient_checkpointing: bool = field(default=True)

# Chatml data formatting for starters
chatml_template = \
    "{% for message in messages %}"\
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"\
        "{% if loop.last and add_generation_prompt %}"\
            "{{ '<|im_start|>assistant\n' }}"\
        "{% endif %}"\
    "{% endfor %}"

class SFTuner:
    def __init__(self, 
                 use_hf: bool = True):
        # Todo: Add option to login to HF hub or not
        pass
            
    def prepare_model(self,
                      model_name: str,
                      **kwargs):
        """
        Prepare config and model according to peft method.
        Currently only works for QLoRA.
        """

        self.llm = model_name
        self.model_args = PeftModelArguments()
        for kwarg in kwargs:
            if hasattr(self.model_args, kwarg):
                setattr(self.model_args, kwarg, kwargs[kwarg])
            else:
                raise AttributeError(f"Invalid model argument: {kwarg}.")

        # if self.model_args.peft_method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # Compute dype for speedup
            bnb_4bit_use_double_quant=True, # Nested quantization
        )

        # lora config
        self.peft_config = LoraConfig(
            lora_alpha= self.model_args.lora_alpha,
            lora_dropout= self.model_args.lora_dropout,
            r=self.model_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.model_args.target_modules
        )

        # Prepare model
        attn = "flash_attention_2" if self.model_args.use_flash_attn else "eager"
        model_dtype = getattr(torch, self.model_args.model_dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation=attn,
            torch_dtype=model_dtype
        )
        
        # Currently preparing tokenizer only for chatml
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm, 
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="<|im_end|>",
            additional_special_tokens=["<|im_start|>user", 
                                       "<|im_start|>assistant",
                                       "<|im_start|>system",
                                       "<|im_end|>",
                                       "<s>", "<pad>"],
            trust_remote_code=True)
        
        self.tokenizer.chat_template = chatml_template
        self.model.resize_token_embeddings(len(self.tokenizer),
                                           pad_to_multiple_of=8)
        
        logging.info(f"Prepared {self.llm} model for training with {self.model_args.peft_method}.")
    
    def preprocess_chat(self, samples):
        conversations = samples['messages']
        batch = [self.tokenizer.apply_chat_template(conv, 
                                                    tokenize=False,) for conv in conversations]
        return {'text': batch}

    def prepare_dataset(self, 
                        stream: bool = False,
                        **kwargs):
        """
        Prepare training (and eval) dataset.
        """
        self.data_args = DataTrainingArguments()

        for kwarg in kwargs:
            if hasattr(self.data_args, kwarg):
                setattr(self.data_args, kwarg, kwargs[kwarg])
            else:
                raise AttributeError(f"Invalid data argument: {kwarg}.")

        if stream:
            raw_datasets = IterableDatasetDict()
            for split in self.data_args.splits.split(","):
                dataset = load_dataset(self.data_args.dataset_name, 
                                    split=split,
                                    streaming=True)
                raw_datasets[split] = dataset
        else:
            raw_datasets = DatasetDict()
            for split in self.data_args.splits.split(","):
                dataset = load_dataset(self.data_args.dataset_name, 
                                    split=split)
                raw_datasets[split] = dataset
        
        raw_datasets = raw_datasets.map(
            self.preprocess_chat,
            remove_columns=raw_datasets["train"].column_names,
            batched=True)

        self.train_split = raw_datasets["train"]
        self.test_split = raw_datasets["test"]
        logging.info(f"Prepared dataset {self.data_args.dataset_name}")
        # logging.info(f"Size of training split: {len(self.train_split)}, \
        #              Size of test split: {len(self.test_split)}")
        
        print(f"Sample text train set: \n{list(self.train_split.take(1))[0]['text']}")

    def train(self, 
              output_dir:str = None, 
              **kwargs):
        if not output_dir:
            output_dir = f"{self.llm.split('/')[-1]}-t2t-sft"
        self.train_args = PeftTrainingArguments(
            output_dir=output_dir)
        for kwarg in kwargs:
            if hasattr(self.train_args, kwarg):
                setattr(self.train_args, kwarg, kwargs[kwarg])
            else:
                raise AttributeError(f"Invalid training argument: {kwarg}.")

        # Gradient checkpointing
        self.model.config.use_cache = not self.train_args.gradient_checkpointing
        if self.train_args.gradient_checkpointing:
            self.train_args.gradient_checkpointing_kwargs = {
                "use_reentrant": self.model_args.use_reentrant
            }

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_split,
            eval_dataset=self.test_split,
            peft_config=self.peft_config,
            packing=False,
            dataset_text_field=self.data_args.text_field,
            max_seq_length=self.data_args.max_seq_length,
            args=self.train_args,
            dataset_kwargs={
                "append_concat_token": self.data_args.append_concat_token,
                "add_special_tokens": self.data_args.add_special_tokens
            })

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