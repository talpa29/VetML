
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
# import evaluate
import pandas as pd
import numpy as np


## 1.0 ##
#########

huggingface_dataset_name = "knkarthick/dialogsum"

dataset = load_dataset(huggingface_dataset_name)

dataset

model_name='google/flan-t5-base'

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# def print_number_of_trainable_model_parameters(model):
#     trainable_model_params = 0
#     all_model_params = 0
#     for _, param in model.named_parameters():
#         all_model_params += param.numel()
#         if param.requires_grad:
#             trainable_model_params += param.numel()
#     return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

# print(print_number_of_trainable_model_parameters(original_model))

# index = 200

# dialogue = dataset['test'][index]['dialogue']
# summary = dataset['test'][index]['summary']

# prompt = f"""
# Summarize the following conversation.

# {dialogue}

# Summary:
# """

# inputs = tokenizer(prompt, return_tensors='pt')
# output = tokenizer.decode(
#     original_model.generate(
#         inputs["input_ids"], 
#         max_new_tokens=200,
#     )[0], 
#     skip_special_tokens=True
# )

dash_line = '-'.join('' for x in range(100))
# print(dash_line)
# print(f'INPUT PROMPT:\n{prompt}')
# print(dash_line)
# print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
# print(dash_line)
# print(f'MODEL GENERATION - ZERO SHOT:\n{output}')



# ## 2.1 ##
# #########

# def tokenize_function(example):
#     start_prompt = 'Summarize the following conversation.\n\n'
#     end_prompt = '\n\nSummary: '
#     prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
#     example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
#     example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
#     return example

# # # The dataset actually contains 3 diff splits: train, validation, test.
# # # The tokenize_function code is handling all data across all splits in batches.
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])

# tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)


# print(f"Shapes of the datasets:")
# print(f"Training: {tokenized_datasets['train'].shape}")
# print(f"Validation: {tokenized_datasets['validation'].shape}")
# print(f"Test: {tokenized_datasets['test'].shape}")

# print(tokenized_datasets)

## 2.2 ##
#########

output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

# training_args = TrainingArguments(
#     output_dir=output_dir,
#     learning_rate=1e-5,
#     num_train_epochs=1,
#     weight_decay=0.01,
#     logging_steps=1,
#     max_steps=1
# )

# trainer = Trainer(
#     model=original_model,
#     args=training_args,
#     train_dataset=tokenized_datasets['train'],
#     eval_dataset=tokenized_datasets['validation']
# )


# trainer.train()
# Use a pipeline as a high-level helper
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_dir = './flan-dialogue-summary-checkpoint'

if os.path.exists(model_dir):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
else:
    print(f"Directory {model_dir} does not exist.")

# Example usage
index = 200
dialogue = dataset['test'][index]['dialogue']
human_baseline_summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

instruct_model_outputs = model.generate(input_ids=input_ids, max_new_tokens=200, num_beams=1)
instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summary}')
print(dash_line)
print(f'INSTRUCT MODEL:\n{instruct_model_text_output}')
