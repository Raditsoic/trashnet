import os
import tensorflow as tf
from huggingface_hub import HfApi, create_repo
import argparse

def push_to_huggingface(model_path, repo_id, path_in_repo, token, commit_message="Update Model"):
    api = HfApi()
    api.set_access_token(token)
    
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Create a temporary directory for SavedModel format
    tmp_save_path = "./tmp_saved_model"
    print("Converting to SavedModel format...")
    model.save(tmp_save_path, save_format='tf')
    
    # Create or get repository
    print(f"Creating/accessing repository: {repo_id}")
    create_repo(repo_id, exist_ok=True, token=token)
    
    if path_in_repo is None:
        print("Uploading model to HuggingFace Hub...")
        api.upload_folder(
            folder_path=tmp_save_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message
        )
        print(f"Model successfully pushed to: https://huggingface.co/{repo_id}")
    else:
        print(f"Uploading model to HuggingFace Hub at {path_in_repo}...")
        api.upload_folder(
            folder_path=tmp_save_path,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=path_in_repo,
            commit_message=commit_message
        )
        print(f"Model successfully pushed to: https://huggingface.co/{repo_id}/{path_in_repo}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Push TensorFlow model to HuggingFace Hub')
    parser.add_argument('--model_path', required=True, help='Path to your saved model')
    parser.add_argument('--repo_id', required=True, help='HuggingFace repository ID (username/model-name)')
    parser.add_argument('--path-in-repo', required=False, default=None, help='Path to save the model in the repository')
    parser.add_argument('--token', required=True, help='HuggingFace API token')
    parser.add_argument('--commit_message', required=False, default="Update Model", help='Commit message for the upload')
    
    args = parser.parse_args()
    
    push_to_huggingface(args.model_path, args.repo_id, args.path_in_repo, args.token, args.commit_message)