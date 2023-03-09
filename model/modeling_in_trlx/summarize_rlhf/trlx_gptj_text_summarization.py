import os
import pathlib
from typing import List

import torch
import trlx
from datasets import load_dataset
from reward_model.reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoTokenizer
from trlx.data.configs import TRLConfig

os.environ['TRANSFORMERS_CACHE'] = "/home/kastanday/.cache/huggingface/transformers"

# USAGE: CUDA_VISIBLE_DEVICES=0,1,2,3 TRANSFORMERS_CACHE=/home/kastanday/.cache/huggingface/transformers accelerate launch --config_file configs/default_accelerate_config.yaml trlx_gptj_text_summarization.py

REWARD_CHECKPOINT_PATH = "reward_model/reward_model_checkpoint/checkpoint-450/pytorch_model.bin"
if not os.path.exists(REWARD_CHECKPOINT_PATH):
  raise ValueError("Reward model checkpoint not found. Please download it from Kastan's huggingface account,"
                   "or train a new one, and place it in the reward_model/reward_model_checkpoint folder.")
  # os.makedirs("reward_model/reward_model_checkpoint", exist_ok=True)
  # os.system(f"wget -O {REWARD_CHECKPOINT_PATH} \
  #       https://huggingface.co/CarperAI/openai_summarize_tldr_reward_model_checkpoint/resolve/main/pytorch_model.bin")
# SFT_MODEL_PATH = "CarperAI/openai_summarize_tldr_sft"
SFT_MODEL_PATH = "kastan/gptj-sft-rlhf"

if __name__ == "__main__":
  # Load the pre-trained reward model
  rw_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
  rw_tokenizer.pad_token = rw_tokenizer.eos_token
  rw_model = GPTRewardModel(SFT_MODEL_PATH)
  rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH))
  rw_model.half()
  rw_model.eval()
  rw_device = torch.device("cuda:{}".format(1))  # set reward model device
  rw_model.to(rw_device)

  def get_scores(samples: List[str]):
    scores_list = []
    batch_size = 2
    for i in range(0, len(samples), batch_size):
      sub_samples = samples[i:i + batch_size]
      sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
      encodings_dict = rw_tokenizer(
          sub_samples,
          truncation=True,
          max_length=config.train.seq_length,
          padding="max_length",
          return_tensors="pt",
      )
      input_ids = encodings_dict["input_ids"].to(rw_device)
      attn_masks = encodings_dict["attention_mask"].to(rw_device)
      input_ids = input_ids.repeat(2, 1)
      attn_masks = attn_masks.repeat(2, 1)
      with torch.no_grad():
        sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
      scores_list.append(sub_scores["chosen_end_scores"])
    scores = torch.cat(scores_list, dim=0)
    return scores

  def get_prompt_dataset(prompts, max_length):
    """
    Get the prompt after T5 decoding to make sure dictionary
    of prompts and summaries is consistent decode prompt from trlX pipeline
    
    "Task: Open book QA. Question: %s \nContext : %s \nAnswer : "
    """
    formatted_prompts = []
    for i in tqdm(range(len(prompts))):
      tmp = tokenizer.decode(
          tokenizer(
              prompts[i],
              truncation=True,
              max_length=max_length - 14,  # to make sure "Question + Answer" dont get truncated
              add_special_tokens=False,
          )["input_ids"],
          skip_special_tokens=True,
      ).strip()
      tmp = "Question: " + tmp + "\nAnswer:"
      tmp = tokenizer.decode(
          tokenizer(tmp, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
          skip_special_tokens=True,
      ).strip()
      formatted_prompts.append(tmp)
    return formatted_prompts

  def reward_fn(samples: List[str], **kwargs):
    # todo: customize the split!!! ---------------------------------------------------
    # we might be good to go!
    original_samples = [text.split("Answer:")[0] + "Answer: " for text in samples]
    original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]
    original_scores = get_scores(original_samples)
    scores = get_scores(samples)
    norms_scores = scores - original_scores
    return norms_scores

  config_path = pathlib.Path(__file__).parent.joinpath("configs/ppo_config_summ_gptj.yml")
  config = TRLConfig.load_yaml(config_path)

  tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)  #, force_download=True)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"
  max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

  # dataset = load_dataset("CarperAI/openai_summarize_tldr")
  # has 'prompt' and 'label' keys
  # https://huggingface.co/datasets/CarperAI/openai_summarize_tldr
  # mine has 'prompt' and 'completion'
  dataset = load_dataset("kastan/rlhf-qa-conditional-generation-v2")

  # Store data into prompt and label pairs
  train_set = [(sample["prompt"], sample["completion"]) for sample in dataset["train"]]
  val_set = [(sample["prompt"], sample["completion"]) for sample in dataset["valid"]]

  # Split contents into summaries and labels
  train_prompt, train_completion = zip(*train_set)
  val_prompts, val_completions = zip(*val_set)

  # Get the OpenAI summaries
  post_summary_dict = {}
  train_prompts = get_prompt_dataset(train_prompt, max_length_input)
  for i in range(len(train_prompts)):
    post_summary_dict[train_prompts[i]] = train_completion[i]
  val_prompts = get_prompt_dataset(val_prompts, max_length_input)
  for i in range(len(val_prompts)):
    post_summary_dict[val_prompts[i]] = val_completions[i]

  trainer = trlx.train(
      reward_fn=reward_fn,
      prompts=train_prompts,
      eval_prompts=val_prompts[0:1000],  # sampling 1000 validation prompts for evaluation speed in training
      config=config,
  )
