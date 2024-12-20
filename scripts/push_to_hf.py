import os
import sys
from huggingface_hub import HfApi, HfRepository

# Arguments: model directory and version
if len(sys.argv) != 3:
    print("Usage: python upload_to_hf.py <model_dir> <version>")
    sys.exit(1)

model_dir = sys.argv[1]
version = sys.argv[2]

USERNAME = "raditsoic"
repo_name = f"model_{version.replace('.', '_')}"  

# Authenticate with Hugging Face
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("Hugging Face token not found. Make sure HF_TOKEN is set.")
    sys.exit(1)

# Create or update the repository
repo_id = f"{USERNAME}/{repo_name}"
print(f"Uploading model version {version} to repository {repo_id}...")

repo = HfRepository(local_dir=model_dir, repo_id=repo_id, use_auth_token=hf_token)
repo.git_add(".")
repo.git_commit(f"Upload version {version}")
repo.git_push()

print(f"Model version {version} successfully uploaded.")