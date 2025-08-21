#!/bin/bash

# Pro Roofing AI - Lambda Labs Setup Script
# Automated setup for Lambda Labs GPU instances

set -e

echo "🚀 Pro Roofing AI - Lambda Labs Setup"
echo "===================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Check if we're on Lambda Labs
print_step "Verifying Lambda Labs environment..."
if nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_status "GPU detected: $GPU_INFO"
else
    print_error "No GPU detected. This script is for Lambda Labs GPU instances."
    exit 1
fi

# Update system packages
print_step "Updating system packages..."
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv htop nvtop curl wget

# Set up project directory
WORK_DIR="$HOME/pro-roofing-ai"
print_step "Setting up project directory at $WORK_DIR..."

if [ -d "$WORK_DIR" ]; then
    print_warning "Project directory already exists. Backing up..."
    mv "$WORK_DIR" "${WORK_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
fi

# Clone repository
print_step "Cloning Pro Roofing AI repository..."
cd $HOME
git clone https://github.com/nexiscart/PRO-X.git pro-roofing-ai
cd pro-roofing-ai

# Create and activate virtual environment
print_step "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
print_step "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
print_step "Installing Python dependencies..."
pip install -r requirements.txt

# Set up environment configuration
print_step "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_status "Environment file created. Please edit with your API keys:"
    print_status "nano .env"
else
    print_status "Environment file already exists"
fi

# Create necessary directories
print_step "Creating directory structure..."
mkdir -p logs/{training,agents,system,performance}
mkdir -p models/{base,checkpoints,final}
mkdir -p data/processed

# Test GPU and PyTorch installation
print_step "Testing GPU and PyTorch installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️  CUDA not available!')
"

# Check available disk space
print_step "Checking disk space..."
DISK_SPACE=$(df -h . | tail -1 | awk '{print $4}')
print_status "Available disk space: $DISK_SPACE"

# Test data processor import
print_step "Testing data processor..."
python3 -c "
try:
    from src.data.enhanced_data_processor import EnhancedRoofingDataProcessor
    print('✅ Data processor imported successfully')
except Exception as e:
    print(f'❌ Data processor import failed: {e}')
"

# Display next steps
echo ""
echo "🎉 Lambda Labs setup completed successfully!"
echo ""
echo -e "${GREEN}Your Pro Roofing AI environment is ready!${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Activate environment: ${BLUE}source venv/bin/activate${NC}"
echo "2. Edit API keys: ${BLUE}nano .env${NC}"
echo "3. Process data: ${BLUE}python src/data/enhanced_data_processor.py${NC}"
echo "4. Start training: ${BLUE}python src/fine_tuning/trainer.py${NC}"
echo ""
echo -e "${GREEN}Monitoring Commands:${NC}"
echo "• GPU usage: ${BLUE}nvtop${NC}"
echo "• System resources: ${BLUE}htop${NC}"
echo "• Training logs: ${BLUE}tail -f logs/training/trainer.log${NC}"
echo ""
echo -e "${YELLOW}Training Configuration:${NC}"
echo "• Model: Qwen3-14B-Instruct"
echo "• Training data: 400K+ roofing examples"
echo "• GPU: $GPU_INFO"
echo "• Location: $WORK_DIR"
echo ""
print_status "Ready to start training! 🚀"