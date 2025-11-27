#!/usr/bin/env python3
"""
Download and prepare NLLB-200 INT8 quantized model for OBS Live Translator
Part 2.2: INT8 Quantization - NLLB-200 (Translation Model)

Model: Xenova/nllb-200-distilled-600M (INT8 variants)
Source: https://huggingface.co/Xenova/nllb-200-distilled-600M
"""

import os
import sys
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

# Model URLs (from Xenova repository)
MODELS_BASE_URL = "https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main"

# NLLB uses encoder-decoder architecture with optional KV caching
FILES_TO_DOWNLOAD = {
    # Encoder (shared across precision variants)
    "onnx/encoder_model.onnx": "nllb-encoder.onnx",
    
    # Decoder INT8 variants
    "onnx/decoder_model_int8.onnx": "nllb-decoder-int8.onnx",
    "onnx/decoder_with_past_model_int8.onnx": "nllb-decoder-with-past-int8.onnx",
    
    # Tokenizer
    "tokenizer.json": "nllb-tokenizer.json",
    "tokenizer_config.json": "nllb-tokenizer-config.json",
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
    print("NLLB-200 INT8 Model Downloader")
    print("Part 2.2: INT8 Quantization for Translation")
    print("=" * 60)
    print()
    
    total_size = 0
    downloaded_files = []
    
    # Download each file
    for source_filename, dest_filename in FILES_TO_DOWNLOAD.items():
        url = f"{MODELS_BASE_URL}/{source_filename}"
        dest_path = models_dir / dest_filename
        
        # Skip if already downloaded
        if dest_path.exists():
            print(f"✓ {dest_filename} already exists, skipping...")
            file_size = dest_path.stat().st_size / (1024 * 1024)  # MB
            total_size += file_size
            continue
        
        # Download
        try:
            download_with_progress(url, dest_path)
            
            # Calculate and display hash
            file_hash = calculate_sha256(dest_path)
            file_size = dest_path.stat().st_size / (1024 * 1024)  # MB
            total_size += file_size
            
            print(f"✓ Downloaded: {dest_filename}")
            print(f"  Size: {file_size:.1f} MB")
            print(f"  SHA256: {file_hash}")
            print()
            
            downloaded_files.append(dest_filename)
            
        except Exception as e:
            print(f"✗ Failed to download {source_filename}: {e}")
            return 1
    
    print()
    print("=" * 60)
    print("✅ Download complete!")
    print("=" * 60)
    print()
    print("Models saved to:", models_dir.absolute())
    print(f"Total size: ~{total_size:.1f} MB")
    print()
    print("Downloaded files:")
    for filename in downloaded_files:
        print(f"  - {filename}")
    print()
    print("Expected performance gains (vs FP16):")
    print("  - Memory: ~50% reduction (2GB → 1GB)")
    print("  - Latency: 1.5-2x faster translation")
    print("  - BLEU score: <3% degradation")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
