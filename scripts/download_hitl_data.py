import os
from huggingface_hub import snapshot_download

def download_from_huggingface(repo_id, local_dir, repo_type="dataset"):
    """
    Downloads a dataset or model from Hugging Face to a specified local directory.

    Args:
        repo_id (str): The Hugging Face repository ID (e.g., "dataset-name" or "model-name").
        local_dir (str): The directory where the dataset/model should be saved.
        repo_type (str): Type of repository, either "dataset" or "model" (default: "dataset").
    """
    os.makedirs(local_dir, exist_ok=True)
    
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        local_dir_use_symlinks=False  # Avoids symlinks for better portability
    )
    
    print(f"Downloaded {repo_id} to {local_dir}")

if __name__ == "__main__":
    # Change these parameters as needed
    repo_id = "huggingface/huihanl/sirius-fleet"  # Example dataset
    local_dir = "data"        # Local directory to save data
    repo_type = "dataset"                    # Change to "model" if downloading a model

    download_from_huggingface(repo_id, local_dir, repo_type)
