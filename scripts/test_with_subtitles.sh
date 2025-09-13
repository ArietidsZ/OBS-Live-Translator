#!/bin/bash

# OBS Live Translator - Test with Subtitle Display
set -e

echo "🚀 OBS Live Translator - Test Suite with Subtitle Display"
echo "========================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is available for test data setup
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}[1/4]${NC} Setting up test data..."
    python3 scripts/download_test_data.py
    echo ""
else
    echo -e "${YELLOW}[WARNING]${NC} Python3 not found. Skipping test data setup."
fi

# Build the project
echo -e "${GREEN}[2/4]${NC} Building the project..."
cargo build --release --bin test-translation 2>/dev/null || {
    echo -e "${YELLOW}[WARNING]${NC} Build failed. Trying with reduced features..."
    cargo build --release --bin test-translation --no-default-features
}
echo ""

# Open browser with subtitle overlay
echo -e "${GREEN}[3/4]${NC} Opening subtitle overlay in browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open "http://localhost:8080" 2>/dev/null || true
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open "http://localhost:8080" 2>/dev/null || true
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    start "http://localhost:8080" 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}[4/4]${NC} Starting translation test with subtitle server..."
echo ""
echo "==============================================="
echo -e "${BLUE}📺 Subtitle Overlay:${NC} http://localhost:8080"
echo -e "${BLUE}🎬 OBS Browser Source Setup:${NC}"
echo "   • URL: http://localhost:8080"
echo "   • Width: 1920"
echo "   • Height: 1080"
echo "   • Custom CSS (optional):"
echo "     body { background-color: transparent; }"
echo "==============================================="
echo ""
echo "Press Ctrl+C to stop the test..."
echo ""

# Run the test translation with subtitle display
RUST_LOG=info cargo run --release --bin test-translation

echo ""
echo "✅ Test completed!"