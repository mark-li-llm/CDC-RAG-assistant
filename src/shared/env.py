"""
Shared environment variable loading module to ensure .env file is loaded from the correct location
"""

import os
from dotenv import load_dotenv

def load_env():
    """
    Load environment variables using absolute path to ensure .env file is loaded from the correct location
    """
    # Get the project root directory path
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(current_dir, '.env')
    print(f"Attempting to load environment variables from: {env_path}")
    
    load_dotenv(env_path)
    
    # Debug information
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        key_preview = f"{openai_api_key[:5]}...{openai_api_key[-4:]}"
        print(f"Loaded OPENAI_API_KEY from {env_path}: {key_preview}")
    else:
        print(f"Warning: Failed to load OPENAI_API_KEY from {env_path}")

    # Return True to indicate successful loading, convenient for the caller to check
    return True 