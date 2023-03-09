from huggingface_hub import login

login()

from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id='kastan/reward_model_checkpoint', exist_ok=True, repo_type='model')

api.upload_folder(
    folder_path="reward_model_checkpoint",  # ../ckpts/best_checkpoint
    commit_message="Step 2 of 3; First attempt at a fine-tuned reward model.",
    repo_id="kastan/reward_model_checkpoint",  # CarperAI/openai_summarize_tldr_ppo
    repo_type="model",
)