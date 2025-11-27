#!/usr/bin/env python3
"""
Download and prepare Parakeet TDT INT8 quantized model for OBS Live Translator
Part 2.1: INT8 Quantization - Parakeet TDT (Medium Profile)

Model: sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8
Source: https://huggingface.co/nvidia/parakeet-tdt-0.6b
"""

import os
import sys
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

# Model URLs (from sherpa-onnx Hugging Face repository)
MODELS_BASE_URL = "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main"

# Parakeet TDT uses a 3-component architecture: encoder + decoder + joiner
FILES_TO_DOWNLOAD = {
    "encoder.int8.onnx": "parakeet-tdt-encoder-int8.onnx",  # Main encoder (652 MB)
    "decoder.int8.onnx": "parakeet-tdt-decoder-int8.onnx",  # Decoder (7.26 MB)
    "joiner.int8.onnx": "parakeet-tdt-joiner-int8.onnx",   # Joiner (1.74 MB)
    "tokens.txt": "parakeet-tdt-tokens.txt",  # Tokenizer vocabulary (9.38 KB)
}

def calculate_sha256(filepath):
    """Calculate SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def download_with_progress(url, dest):
    """Download file with progress bar"""
    def reporthook(count, block_size, total_size):
        percent = min(int(count * block_size * 100 / total_size), 100)
        sys.stdout.write(f"\r...{percent}% complete")
        sys.stdout.flush()
    
    print(f"Downloading {os.path.basename(dest)}...")
    urlretrieve(url, dest, reporthook=reporthook)
    print()  # New line after progress

def main():
    # Determine output directory
    script_dir = Path(__file__).parent.parent
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Parakeet TDT INT8 Model Downloader")
    print("Part 2.1: INT8 Quantization for Medium Profile")
    print("=" * 60)
    print()
    
    # Download each file
    for source_filename, dest_filename in FILES_TO_DOWNLOAD.items():
        url = f"{MODELS_BASE_URL}/{source_filename}"
        dest_path = models_dir / dest_filename
        
        # Skip if already downloaded
        if dest_path.exists():
            print(f"✓ {dest_filename} already exists, skipping...")
            continue
        
        # Download
        try:
            download_with_progress(url, dest_path)
            
            # Calculate and display hash
            file_hash = calculate_sha256(dest_path)
            file_size = dest_path.stat().st_size / (1024 * 1024)  # MB
            
            print(f"✓ Downloaded: {dest_filename}")
            print(f"  Size: {file_size:.1f} MB")
            print(f"  SHA256: {file_hash}")
            print()
            
        except Exception as e:
            print(f"✗ Failed to download {source_filename}: {e}")
            return 1
    
    print()
    print("=" * 60)
    print("✅ Download complete!")
    print("=" * 60)
    print()
    print("Models saved to:", models_dir.absolute())
    print()
    print("Expected performance gains (vs FP16):")
    print("  - Memory: ~50% reduction (1.2GB → 600MB)")
    print("  - Latency: 1.5-2x faster inference")
    print("  - Accuracy: <5% WER degradation")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
