import os
import sys
import logging
from typing import Optional
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from datasets import load_dataset, Dataset, interleave_datasets
from accelerate import Accelerator
from transformers import HfArgumentParser, TrainingArguments
from utils import load_rehearsal_dataset

# Suppress advisory warnings and set environment variables
os.environ['TRANSFORMERS_NO_ADVISORY_WARNING'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None, metadata={"help": "Path to the base model"})
    load_in_8bit: bool = field(default=False, metadata={"help": "Load model in 8-bit mode"})
    use_lora: bool = field(default=False, metadata={"help": "Use LoRA quantization"})
    lora_r: int = field(default=8, metadata={"help": "LoRA r parameter"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout rate"})
    lora_target_modules: str = field(default="q, v", metadata={"help": "LoRA target modules"})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the data directory"})
    source_max_length: int = field(default=256, metadata={"help": "Max length of source sequences"})
    model_max_length: int = field(default=512, metadata={"help": "Max length of target sequences"})
    preprocessing_num_workers: int = field(default=4, metadata={"help": "Number of preprocessing workers"})
    continual_size: int = field(default=1000, metadata={"help": "Number of examples for experience replay"})
    val_set_size: float = field(default=0.2, metadata={"help": "Validation set proportion"})

@dataclass
class LiusTrainingArguments(TrainingArguments):
    wandb_project: Optional[str] = field(default="lius_project", metadata={"help": "WandB project name"})
    push_to_hub: bool = field(default=False, metadata={"help": "Push model to Hugging Face Hub"})

def load_model_and_tokenizer(model_args):
    quant_config = BitsAndBytesConfig(load_in_8bit=model_args.load_in_8bit)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=quant_config)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    return model, tokenizer

def prepare_datasets(data_args, tokenizer):
    raw_datasets = load_dataset(data_args.data_path, token=os.getenv("HF_TOKEN"))
    rehearsal_data = load_rehearsal_dataset(n_samples=data_args.continual_size)
    rehearsal_dataset = Dataset.from_list(rehearsal_data)
    raw_datasets["train"] = interleave_datasets([rehearsal_dataset, raw_datasets["train"]], stopping_strategy="all_exhausted")

    def tokenize_function(data_point):
        full_prompt = f'{data_point["input"]} {data_point["output"]}'
        user_prompt = f'{data_point["input"]}'
        user_prompt_len = len(tokenizer(user_prompt, truncation=True, max_length=data_args.model_max_length)["input_ids"])
        tokenized_full_prompt = tokenizer(full_prompt + tokenizer.eos_token, truncation=True, max_length=data_args.model_max_length)
        tokenized_full_prompt["labels"] = [IGNORE_INDEX] * user_prompt_len + tokenized_full_prompt["input_ids"].copy()[user_prompt_len:]
        tokenized_full_prompt.pop('attention_mask')
        return tokenized_full_prompt

    train_val_split = raw_datasets["train"].train_test_split(test_size=data_args.val_set_size, seed=42)
    train_dataset = train_val_split["train"].map(tokenize_function, num_proc=data_args.preprocessing_num_workers, remove_columns=train_val_split["test"].column_names, desc="Tokenizing train dataset")
    val_dataset = train_val_split["test"].map(tokenize_function, num_proc=data_args.preprocessing_num_workers, remove_columns=train_val_split["test"].column_names, desc="Tokenizing validation dataset")

    return train_dataset, val_dataset

def setup_model_for_training(model, model_args):
    if model_args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model.enable_input_require_grads()

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules.split(","),
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model

def train():
    accelerator = Accelerator()

    parser = HfArgumentParser((ModelArguments, DataArguments, LiusTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    setup_logging(training_args)

    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Training arguments: {training_args}")

    model, tokenizer = load_model_and_tokenizer(model_args)
    model = setup_model_for_training(model, model_args)

    train_dataset, val_dataset = prepare_datasets(data_args, tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    model.config.use_cache = False

    if model_args.use_lora:
        original_state_dict = model.state_dict()
        model.state_dict = lambda self, *_, **__: get_peft_model_state_dict(self, original_state_dict)

    trainer.train()

    model.save_pretrained(
        training_args.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )

if __name__ == "__main__":
    train()



