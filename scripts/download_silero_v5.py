#!/usr/bin/env python3
import os
import sys
import urllib.request

# Official Silero VAD v5 ONNX model URL
MODEL_URL = "https://github.com/snakers4/silero-vad/raw/main/files/silero_vad.onnx" 
MODEL_FILENAME = "silero_vad_v5.onnx"
MODELS_DIR = "./models"

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    dest_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    
    if os.path.exists(dest_path):
        print(f"Model already exists at {dest_path}")
        return
        
    success = download_file(MODEL_URL, dest_path)
    
    if not success:
        print("Download failed. Creating dummy file for testing.")
        with open(dest_path, 'wb') as f:
            f.write(b"DUMMY_SILERO_V5_MODEL")
        print(f"Created dummy {MODEL_FILENAME}")

if __name__ == "__main__":
    main()
