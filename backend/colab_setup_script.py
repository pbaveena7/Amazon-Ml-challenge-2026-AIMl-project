import os
import subprocess
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from app.main import app

def setup_colab(ngrok_auth_token=None):
    """
    Sets up the FastAPI server in Google Colab and exposes it via ngrok.
    """
    if ngrok_auth_token:
        ngrok.set_auth_token(ngrok_auth_token)
        
    # Start ngrok tunnel
    public_url = ngrok.connect(8000).public_url
    print(f"==================================================")
    print(f"API is exposed at: {public_url}")
    print(f"Paste this URL in the frontend API Connection page.")
    print(f"==================================================")
    
    # Allow nested asyncio loops in Colab
    nest_asyncio.apply()
    
    # Run Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Get token from env or prompt
    token = os.environ.get("NGROK_AUTH_TOKEN", "")
    setup_colab(token)
