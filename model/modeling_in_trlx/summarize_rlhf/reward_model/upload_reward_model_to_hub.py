from huggingface_hub import login

login()

from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./reward_model_checkpoint",
    commit_message="Step 2 of 3; First attempt at reward model.",
    repo_id="kastan/reward_model_checkpoint",
    repo_type="model",
)