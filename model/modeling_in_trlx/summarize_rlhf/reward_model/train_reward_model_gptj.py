import os

import torch
from datasets import load_dataset
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

#### RUN ME USING:  CUDA_VISIBLE_DEVICES=0 deepspeed train_reward_model_gptj.py
#### OR maybe (untested):    CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=1 train_reward_model_gptj.py


def create_comparison_dataset(path="kastan/rlhf-qa-conditional-generation-v2", split="train"):
  dataset = load_dataset(path, split=split)
  pairs = []
  for sample in tqdm(dataset):
    pair = {}
    # prompt = sample["prompt"]
    prompt = sample["Question"]  # updated to kastan's dataset
    chosen_summary = sample["Chosen"]
    rejected_summary = sample["Rejected"]
    if chosen_summary == rejected_summary:
      continue
    if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
      continue
    pair["chosen"] = prompt + "\n" + chosen_summary
    pair["rejected"] = prompt + "\n" + rejected_summary
    pairs.append(pair)
  return pairs


class PairwiseDataset(Dataset):

  def __init__(self, pairs, tokenizer, max_length):
    self.chosen_input_ids = []
    self.chosen_attn_masks = []
    self.rejected_input_ids = []
    self.rejected_attn_masks = []
    for pair in tqdm(pairs):
      chosen, rejected = pair["chosen"], pair["rejected"]
      chosen_encodings_dict = tokenizer(
          "<|startoftext|>" + chosen + "<|endoftext|>",
          truncation=True,
          max_length=max_length,
          padding="max_length",
          return_tensors="pt",
      )
      rejected_encodings_dict = tokenizer(
          "<|startoftext|>" + rejected + "<|endoftext|>",
          truncation=True,
          max_length=max_length,
          padding="max_length",
          return_tensors="pt",
      )
      self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
      self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
      self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
      self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

  def __len__(self):
    return len(self.chosen_input_ids)

  def __getitem__(self, idx):
    return (
        self.chosen_input_ids[idx],
        self.chosen_attn_masks[idx],
        self.rejected_input_ids[idx],
        self.rejected_attn_masks[idx],
    )


class DataCollatorReward:

  def __call__(self, data):
    batch = {}
    batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
    batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
    batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
    return batch


def compute_metrics(eval_preds):
  chosen_end_scores = eval_preds.predictions[0]  # chosen scores
  rejected_end_scores = eval_preds.predictions[1]  # rejected scores

  result = {}
  acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
  result["accuracy"] = acc

  return result


if __name__ == "__main__":
  tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
  tokenizer.pad_token = tokenizer.eos_token

  # output_dir = "/home/kastanday/reward_model_checkpoint"
  output_dir = "reward_model_checkpoint"
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  training_args = TrainingArguments(
      output_dir=output_dir,
      num_train_epochs=5,
      logging_steps=10,
      gradient_accumulation_steps=1,
      save_strategy="steps",
      save_steps=450,
      save_total_limit=1,
      evaluation_strategy="steps",
      per_device_train_batch_size=1,
      per_device_eval_batch_size=2,
      eval_accumulation_steps=1,
      eval_steps=500,
      warmup_steps=100,
      logging_dir="./logs",
      fp16=True,  # maybe switch to FP32, not convinced this will work. 
      bf16=False,  # not supported on V100. 
      learning_rate=1e-5,
      deepspeed="./ds_config_gpt_j.json",
      push_to_hub=True,
      hub_strategy='end',
      # max_steps=20,  # for testing
      hub_token="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",
  )

  # Initialize the reward model from the (supervised) fine-tuned GPT-J
  model = GPTRewardModel("kastan/gptj-sft-rlhf")

  # Freeze the first 70% of the hidden layers of the reward model backbone
  layers = model.transformer.h
  num_layers = len(layers)
  num_unfrozen = int(0.3 * num_layers)
  for layer in layers[:-num_unfrozen]:
    layer.requires_grad_(False)

  # Create the comparisons datasets
  # data_path = "CarperAI/openai_summarize_comparisons"
  data_path = "kastan/rlhf-qa-comparisons"
  train_pairs = create_comparison_dataset(data_path, "train")
  #   val_pairs = create_comparison_dataset(data_path, "train")
  val_pairs = create_comparison_dataset(data_path, "train")  # todo change back

  # Make pairwise datasets for training
  max_length = 550
  train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
  val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

  # Create the collator to gather batches of pairwise comparisons
  data_collator = DataCollatorReward()

  Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      compute_metrics=compute_metrics,
      eval_dataset=val_dataset,
      data_collator=data_collator,
  ).train()
