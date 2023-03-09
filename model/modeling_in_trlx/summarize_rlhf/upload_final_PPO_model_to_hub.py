from huggingface_hub import login

login()

from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id='kastan/rlhf-qa-ppo', exist_ok=True, repo_type='model')

api.upload_folder(
    folder_path="../ckpts/best_checkpoint",  # ../ckpts/best_checkpoint
    commit_message="Step 3 of 3; First attempt at a PPO fine-tuned model.",
    repo_id="kastan/rlhf-qa-ppo",  # CarperAI/openai_summarize_tldr_ppo
    repo_type="model",
)