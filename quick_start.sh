#!/bin/bash

# Pro Roofing AI - Complete Quick Start
# One-command setup and training launcher

set -e

echo "🏠 Pro Roofing AI - Complete Quick Start"
echo "======================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}📍 $1${NC}"
}

print_success() {
    echo -e "${PURPLE}✨ $1${NC}"
}

# Check if running in correct directory
if [ ! -f "README.md" ] || [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the pro-roofing-ai root directory"
    exit 1
fi

print_step "Checking system requirements..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Python version: $PYTHON_VERSION"

# Check for CUDA availability
print_step "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_success "NVIDIA GPU detected: $GPU_INFO"
    GPU_AVAILABLE=true
else
    print_warning "No NVIDIA GPU detected. Training will be slow on CPU."
    GPU_AVAILABLE=false
fi

# Create and activate virtual environment
print_step "Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

source venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with appropriate backend
print_step "Installing PyTorch..."
if [ "$GPU_AVAILABLE" = true ]; then
    print_status "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    print_status "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install requirements
print_step "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_step "Creating directory structure..."
mkdir -p logs/{training,agents,system,performance}
mkdir -p models/{base,checkpoints,final}
mkdir -p data/{processed,nrca_manuals}

# Setup environment file
print_step "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_status "Environment file created from template"
    print_warning "🔑 IMPORTANT: Edit .env file with your API keys before training!"
    echo "   Required: OPENAI_API_KEY, WANDB_API_KEY, HUGGINGFACE_TOKEN"
else
    print_status "Environment file already exists"
fi

# Check if AI Drive data is available
print_step "Checking for roofing datasets..."
AI_DRIVE_PATH="/mnt/aidrive"
DATA_AVAILABLE=false

if [ -d "$AI_DRIVE_PATH" ]; then
    JSON_COUNT=$(find "$AI_DRIVE_PATH" -name "*.json" | wc -l)
    PDF_COUNT=$(find "$AI_DRIVE_PATH" -name "*.pdf" | wc -l)
    
    if [ $JSON_COUNT -gt 0 ]; then
        print_success "Found $JSON_COUNT JSON datasets in AI Drive"
        DATA_AVAILABLE=true
        
        # Copy datasets to local directory
        print_step "Copying datasets from AI Drive..."
        for json_file in "$AI_DRIVE_PATH"/*.json; do
            if [ -f "$json_file" ]; then
                filename=$(basename "$json_file")
                print_status "Copying $filename..."
                cp "$json_file" "data/raw/"
            fi
        done
        
        # Copy PDF manuals if available
        if [ $PDF_COUNT -gt 0 ]; then
            print_status "Found $PDF_COUNT PDF manuals - copying to nrca_manuals/"
            for pdf_file in "$AI_DRIVE_PATH"/*.pdf; do
                if [ -f "$pdf_file" ]; then
                    cp "$pdf_file" "data/nrca_manuals/"
                fi
            done
            
            # Also check for manual directory
            if [ -d "$AI_DRIVE_PATH/Roofing Data Manuals" ]; then
                print_status "Copying manuals from Roofing Data Manuals directory..."
                cp "$AI_DRIVE_PATH/Roofing Data Manuals"/*.pdf "data/nrca_manuals/" 2>/dev/null || true
            fi
        fi
        
        print_success "✅ All datasets copied successfully!"
    fi
else
    print_warning "AI Drive not accessible. Using existing data in data/raw/"
    if [ -d "data/raw" ] && [ "$(ls -A data/raw)" ]; then
        DATA_AVAILABLE=true
        print_status "Found existing data in data/raw directory"
    fi
fi

# Validate data availability
if [ "$DATA_AVAILABLE" = false ]; then
    print_error "No training data found! Please ensure datasets are available."
    exit 1
fi

# Test installation
print_step "Testing installation..."
python -c "
import torch
import transformers
import datasets
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ Transformers version: {transformers.__version__}')
print(f'✅ Datasets version: {datasets.__version__}')

if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️  CUDA not available - using CPU')
"

# Test core modules
print_status "Testing core modules..."
python -c "
from src.data.enhanced_data_processor import EnhancedRoofingDataProcessor
from src.fine_tuning.trainer import RoofingAITrainer
print('✅ All core modules imported successfully')
"

# Process training data
print_step "Processing training data..."
print_status "This will process all 400K+ examples from JSON files and manuals..."
python src/data/enhanced_data_processor.py

if [ $? -eq 0 ]; then
    print_success "✅ Data processing completed successfully!"
else
    print_error "❌ Data processing failed"
    exit 1
fi

# Display setup summary
echo ""
echo "🎉 Pro Roofing AI Setup Complete!"
echo "================================="
echo ""
echo -e "${GREEN}Setup Summary:${NC}"
echo "• Environment: Ready ✅"
echo "• Dependencies: Installed ✅"
echo "• Training Data: Processed ✅"
echo "• Model: Qwen3-14B-Instruct configured ✅"
echo "• GPU: $([ "$GPU_AVAILABLE" = true ] && echo "Available ✅" || echo "Not available ⚠️")"
echo ""
echo -e "${YELLOW}Training Configuration:${NC}"
echo "• Model: Qwen3-14B-Instruct"
echo "• Training Examples: 400K+"
echo "• Fine-tuning: LoRA with 4-bit quantization"
echo "• Expected Duration: 6-10 hours (GPU) / 24+ hours (CPU)"
echo ""
echo -e "${BLUE}Ready to Start Training!${NC}"
echo ""
echo -e "${GREEN}Option 1 - Manual Training:${NC}"
echo "  1. Edit API keys: ${YELLOW}nano .env${NC}"
echo "  2. Start training: ${YELLOW}python src/fine_tuning/trainer.py${NC}"
echo ""
echo -e "${GREEN}Option 2 - Lambda Labs Auto-Setup:${NC}"
echo "  ${YELLOW}bash deployment/lambda_labs/launch_training.sh${NC}"
echo ""
echo -e "${GREEN}Monitoring Commands:${NC}"
echo "• GPU usage: ${YELLOW}nvtop${NC} or ${YELLOW}nvidia-smi${NC}"
echo "• Training logs: ${YELLOW}tail -f logs/training/trainer.log${NC}"
echo "• System resources: ${YELLOW}htop${NC}"
echo ""
echo -e "${PURPLE}Important Notes:${NC}"
echo "🔑 Remember to set your API keys in .env before training"
echo "💾 Final model will be saved to: models/final/"
echo "📊 Monitor training progress in WandB dashboard"
echo "⏰ Training will take several hours - use screen/tmux for long sessions"
echo ""
print_success "Ready to train your roofing AI expert! 🚀🏠"