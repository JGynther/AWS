import torch
from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    AutoPeftModelForCausalLM,
)


MODEL_ID = "tiiuae/falcon-7b"


# Bitsandbytes config for 4-bit training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,  # NOTE: allows for remote code execution
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer.pad_token = tokenizer.eos_token


# LoRA config
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()


# Loading dataset specified with estimator
dataset = load_from_disk("/opt/ml/input/data/training")
# Shuffle data and select 15k random samples
dataset = dataset.shuffle().select(range(15000))


# Training arguments
# This should be done by setting hyperparameters and not hard coded like here
# Read more: https://huggingface.co/docs/sagemaker/train#prepare-a-transformers-finetuning-script
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    save_strategy="no",
    output_dir="/tmp",
    logging_dir="/tmp/logs",
    logging_steps=10,
)

# Start training
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
trainer.model.save_pretrained("/tmp", safe_serialization=False)


# Clear GPU memory
del model
del trainer
torch.cuda.empty_cache()


# Merge LoRA weights with model
# Read more: https://huggingface.co/docs/peft/conceptual_guides/lora#merge-lora-weights-into-the-base-model
model = AutoPeftModelForCausalLM.from_pretrained(
    "/tmp",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,  # NOTE: allows for remote code execution
)
merged = model.merge_and_unload()
merged.save_pretrained("/opt/ml/model/", safe_serialization=True)


# Save tokenizer for easier inference
tokenizer.save_pretrained("/opt/ml/model/")
