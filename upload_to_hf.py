#!/usr/bin/env python3
"""
Script to upload a model to Hugging Face Hub
Usage: python upload_to_hf.py --model_path /path/to/model --repo_id username/model-name

Create a .env file in the same directory with:
HF_TOKEN=your_huggingface_token_here
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, login
from huggingface_hub.utils import HfHubHTTPError


try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Falling back to system environment variables...")

def main():
    parser = argparse.ArgumentParser(description="Upload a model to Hugging Face Hub")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the model directory containing model files"
    )
    parser.add_argument(
        "--repo_id", 
        type=str, 
        required=True,
        help="Hugging Face repository ID (format: username/model-name)"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Make the repository private (default: public)"
    )
    parser.add_argument(
        "--commit_message", 
        type=str, 
        default="Upload model",
        help="Commit message for the upload"
    )
    
    args = parser.parse_args()
    
    # Check if model path exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist!")
        sys.exit(1)
    
    if not model_path.is_dir():
        print(f"Error: Model path {model_path} is not a directory!")
        sys.exit(1)
    
    # Check for required model files
    required_files = ["config.json"]
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: Missing files {missing_files} in model directory")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Get token from environment (loaded from .env file or system environment)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("Error: No Hugging Face token found!")
        print("Option 1: Create a .env file with: HF_TOKEN=your_token_here")
        print("Option 2: Set environment variable: export HF_TOKEN=your_token_here")
        print("Get your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    try:
        # Login with token
        print("Logging in to Hugging Face...")
        login(token=token)
        
        # Initialize API
        api = HfApi()
        
        # Create repository
        print(f"Creating repository {args.repo_id}...")
        try:
            api.create_repo(
                repo_id=args.repo_id, 
                exist_ok=True,
                private=args.private,
                repo_type="model"
            )
            print(f"Repository {args.repo_id} created/verified successfully")
        except HfHubHTTPError as e:
            if "already exists" in str(e):
                print(f"Repository {args.repo_id} already exists, continuing...")
            else:
                raise e
        
        # Upload model files
        print(f"Uploading model from {model_path}...")
        print("This might take a while for large models...")
        
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
            ignore_patterns=[".git", "*.pyc", "__pycache__", ".DS_Store"]
        )
        
        print(f"✅ Model uploaded successfully!")
        print(f"🔗 View your model at: https://huggingface.co/{args.repo_id}")
        
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()