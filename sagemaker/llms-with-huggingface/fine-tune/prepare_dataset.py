import sagemaker
from datasets import load_dataset
from transformers import AutoTokenizer

session = sagemaker.Session()
role = sagemaker.get_execution_role()


dataset = load_dataset("code_search_net", "all", split="train")


def format(sample):
    code = f"### Input:\n{sample['func_code_string']}"
    comment = f"### Response:\n{sample['func_documentation_string']}"
    return code + "\n\n" + comment


model = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.pad_token = tokenizer.eos_token


def template(sample):
    sample["text"] = f"{format(sample)}{tokenizer.eos_token}"
    return sample


dataset = dataset.map(template, remove_columns=list(dataset.features))
lm_dataset = dataset.map(
    lambda sample: tokenizer(sample["text"], truncation=True),
    remove_columns=list(dataset.features),
    batched=True
)

lm_dataset.save_to_disk(f"s3://{session.default_bucket()}/processed/train")