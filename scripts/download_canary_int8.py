#!/usr/bin/env python3
import os
import sys
import urllib.request

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        sys.exit(1)

def download_canary_int8():
    base_url = "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/resolve/main"
    output_dir = "models"
    
    os.makedirs(output_dir, exist_ok=True)
    
    files = [
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "tokens.txt",
    ]
    
    for filename in files:
        url = f"{base_url}/{filename}"
        dst_name = f"canary-180m-{filename}"
        dst = os.path.join(output_dir, dst_name)
        download_file(url, dst)
            
    print("Download complete.")

if __name__ == "__main__":
    download_canary_int8()
