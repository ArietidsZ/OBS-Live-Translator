#!/bin/bash
# High-Performance Model Download Script
set -e

echo "🧠 Downloading AI models for OBS Live Translator v2.0"

# Create models directory
mkdir -p models

# Define model configurations
declare -A MODELS=(
    ["whisper-tiny"]="openai/whisper-tiny"
    ["whisper-small"]="openai/whisper-small" 
    ["whisper-base"]="openai/whisper-base"
    ["whisper-large-v3"]="openai/whisper-large-v3"
    ["nllb-600m"]="facebook/nllb-200-distilled-600M"
    ["nllb-1.3b"]="facebook/nllb-200-1.3B"
    ["sensevoice"]="FunAudioLLM/SenseVoiceSmall"
)

# Performance mode selection
show_usage() {
    echo "Usage: ./download-models.sh [MODE]"
    echo ""
    echo "Modes:"
    echo "  gaming     - Minimal models for ultra-low latency (1GB)"
    echo "  balanced   - Optimal quality/performance balance (4GB)"
    echo "  quality    - Maximum accuracy models (8GB)"
    echo "  all        - Download all models (15GB)"
    echo "  custom     - Interactive model selection"
    echo ""
    exit 1
}

download_model() {
    local model_name=$1
    local repo_id=$2
    
    echo "📥 Downloading $model_name..."
    
    # Create model directory
    mkdir -p "models/$model_name"
    
    # Download using huggingface_hub if available, otherwise use curl
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$repo_id" --local-dir "models/$model_name" --local-dir-use-symlinks False
    else
        echo "⚠️  huggingface-cli not found, using direct download..."
        
        # Download key files
        curl -L "https://huggingface.co/$repo_id/resolve/main/config.json" -o "models/$model_name/config.json" 2>/dev/null || true
        curl -L "https://huggingface.co/$repo_id/resolve/main/pytorch_model.bin" -o "models/$model_name/pytorch_model.bin" 2>/dev/null || true
        curl -L "https://huggingface.co/$repo_id/resolve/main/tokenizer.json" -o "models/$model_name/tokenizer.json" 2>/dev/null || true
        curl -L "https://huggingface.co/$repo_id/resolve/main/model.safetensors" -o "models/$model_name/model.safetensors" 2>/dev/null || true
    fi
    
    echo "✅ $model_name downloaded"
}

download_gaming_mode() {
    echo "🎮 Gaming Mode: Ultra-low latency setup"
    download_model "whisper-tiny" "${MODELS[whisper-tiny]}"
    download_model "nllb-600m" "${MODELS[nllb-600m]}"
    echo "🎯 Gaming mode setup complete (target: <25ms latency)"
}

download_balanced_mode() {
    echo "⚖️  Balanced Mode: Optimal quality/performance"
    download_model "whisper-small" "${MODELS[whisper-small]}"
    download_model "nllb-600m" "${MODELS[nllb-600m]}"
    download_model "sensevoice" "${MODELS[sensevoice]}"
    echo "🎯 Balanced mode setup complete (target: 50-100ms latency)"
}

download_quality_mode() {
    echo "🏆 Quality Mode: Maximum accuracy"
    download_model "whisper-base" "${MODELS[whisper-base]}"
    download_model "whisper-large-v3" "${MODELS[whisper-large-v3]}"
    download_model "nllb-1.3b" "${MODELS[nllb-1.3b]}"
    download_model "sensevoice" "${MODELS[sensevoice]}"
    echo "🎯 Quality mode setup complete (target: maximum accuracy)"
}

download_all_models() {
    echo "📦 Downloading all models..."
    for model_name in "${!MODELS[@]}"; do
        download_model "$model_name" "${MODELS[$model_name]}"
    done
    echo "🎯 All models downloaded"
}

interactive_selection() {
    echo "🛠️  Interactive Model Selection"
    echo ""
    echo "Available models:"
    
    local i=1
    local selected_models=()
    
    for model_name in "${!MODELS[@]}"; do
        echo "  $i) $model_name (${MODELS[$model_name]})"
        ((i++))
    done
    
    echo ""
    echo "Enter model numbers separated by spaces (e.g., 1 3 5): "
    read -r selections
    
    local model_array=($(printf '%s\n' "${!MODELS[@]}" | sort))
    
    for selection in $selections; do
        if [[ $selection -ge 1 && $selection -le ${#MODELS[@]} ]]; then
            local idx=$((selection - 1))
            local model_name="${model_array[$idx]}"
            download_model "$model_name" "${MODELS[$model_name]}"
        fi
    done
}

check_disk_space() {
    local required_gb=$1
    local available_gb
    
    if command -v df &> /dev/null; then
        available_gb=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
        
        if [[ $available_gb -lt $required_gb ]]; then
            echo "⚠️  Warning: Only ${available_gb}GB available, ${required_gb}GB recommended"
            echo "Continue anyway? (y/N): "
            read -r continue_download
            
            if [[ ! $continue_download =~ ^[Yy]$ ]]; then
                echo "❌ Download cancelled"
                exit 1
            fi
        fi
    fi
}

verify_downloads() {
    echo "🔍 Verifying downloaded models..."
    
    local total_files=0
    local valid_files=0
    
    for model_dir in models/*/; do
        if [[ -d "$model_dir" ]]; then
            local model_name=$(basename "$model_dir")
            echo "Checking $model_name..."
            
            for file in "$model_dir"/*; do
                if [[ -f "$file" && -s "$file" ]]; then
                    ((valid_files++))
                fi
                ((total_files++))
            done
        fi
    done
    
    echo "✅ Verification complete: $valid_files/$total_files files valid"
    
    if [[ $valid_files -eq $total_files ]]; then
        echo "🎉 All models downloaded successfully!"
    else
        echo "⚠️  Some files may be incomplete. Run script again if needed."
    fi
}

# Main execution
case "${1:-balanced}" in
    "gaming")
        check_disk_space 2
        download_gaming_mode
        ;;
    "balanced")
        check_disk_space 5
        download_balanced_mode
        ;;
    "quality") 
        check_disk_space 10
        download_quality_mode
        ;;
    "all")
        check_disk_space 15
        download_all_models
        ;;
    "custom")
        interactive_selection
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo "❌ Unknown mode: $1"
        show_usage
        ;;
esac

verify_downloads

echo ""
echo "🎯 Model download complete!"
echo "💡 Tip: Use 'cargo run --release --bin obs-translator-server -- --benchmark' to test performance"