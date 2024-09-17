import os
import wandb
import random
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from accelerate import Accelerator

os.environ["WANDB_PROJECT"] = "TravelSurvey-Sim-LLM"

wandb.login(key='ENTER KEY HERE')


# Load model and tokenizer

model_name = 'meta-llama/Llama-2-70b-hf'
access_token = open('access_token').read()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=access_token,
    load_in_8bit=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 2
model = prepare_model_for_kbit_training(model)
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Apply Lora
lora_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.1,
    r=256,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())
print(model)

# Load Dataset
train_data = load_dataset(
    'csv',
    data_files=(
        'training_datasets/training_dataset_large_exclude_inf_cities.csv'
    )
)

text_column = "context"
label_column = "output"
max_length = 1024


def preprocess_function(examples):
    batch_size = len(examples[text_column])
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(examples[text_column])
    labels = tokenizer(targets, add_special_tokens=False)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + \
            label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(
            model_inputs["input_ids"][i]
        )
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (
            max_length - len(sample_input_ids)
        ) + model_inputs["attention_mask"][i]
        labels["input_ids"][i] = [-100] * (
            max_length - len(sample_input_ids)
        ) + label_input_ids
        model_inputs["input_ids"][i] = model_inputs[
            "input_ids"
        ][i][:max_length]
        model_inputs["attention_mask"][i] = model_inputs[
            "attention_mask"
        ][i][:max_length]
        labels["input_ids"][i] = labels["input_ids"][i][:max_length]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


processed_datasets = train_data.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=train_data["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]


training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    logging_steps=2,
    learning_rate=1e-4,
    max_grad_norm=0.3,
    num_train_epochs=5,
    warmup_ratio=0.05,
    save_strategy="epoch",
    group_by_length=True,
    output_dir="model_output_large_exclude_inf_cities",
    report_to="wandb",
    lr_scheduler_type="cosine",
)
print(training_args.device)

accelerator = Accelerator()
trainer = accelerator.prepare(Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=default_data_collator,
))
trainer.train()


i = random.randint(0, len(train_data["train"])-1)
context = train_data["train"][i]["context"]

batch = tokenizer(context, return_tensors="pt")
batch = {k: v.to("cuda") for k, v in batch.items()}
model.eval()
output_tokens = model.generate(
    **batch,
    max_new_tokens=256,
    do_sample=True,
    temperature=1,
    top_p=0.9,
    top_k=50,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
target_predicted = tokenizer.decode(
    output_tokens[0],
    skip_special_tokens=False
)
print(f"{context=} \n\n {target_predicted=} \n\n ")
