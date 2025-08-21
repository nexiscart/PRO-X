#!/bin/bash

# Pro Roofing AI Environment Setup Script
# This script sets up the complete development environment

set -e  # Exit on any error

echo "🏠 Pro Roofing AI Environment Setup"
echo "=================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if Python version is 3.8 or higher
if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    print_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Check for CUDA availability
print_step "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "No NVIDIA GPU detected. Training will use CPU (not recommended)"
fi

# Create virtual environment
print_step "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_step "Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch with CUDA support if available
print_step "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
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
mkdir -p data/processed

print_status "Directory structure created"

# Setup environment file
print_step "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_status "Environment file created from template"
    print_warning "Please edit .env file with your API keys and configuration"
else
    print_status "Environment file already exists"
fi

# Initialize git hooks (if git is available)
if command -v git &> /dev/null && [ -d ".git" ]; then
    print_step "Setting up git hooks..."
    
    # Create pre-commit hook for code quality
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for Pro Roofing AI

echo "Running pre-commit checks..."

# Check Python syntax
if ! python -m py_compile src/**/*.py; then
    echo "Python syntax errors found!"
    exit 1
fi

# Run basic linting (if available)
if command -v flake8 &> /dev/null; then
    flake8 src/ --max-line-length=100 --ignore=E203,W503
fi

echo "Pre-commit checks passed!"
EOF
    
    chmod +x .git/hooks/pre-commit
    print_status "Git hooks configured"
fi

# Test installation
print_step "Testing installation..."

# Test basic imports
python -c "
import torch
import transformers
import datasets
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ Transformers version: {transformers.__version__}')
print(f'✅ Datasets version: {datasets.__version__}')

if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'✅ CUDA version: {torch.version.cuda}')
else:
    print('⚠️  CUDA not available')
"

# Test data processing
print_status "Testing data processor..."
python -c "
from src.data.enhanced_data_processor import EnhancedRoofingDataProcessor
config = {'data': {'max_length': 4096}}
processor = EnhancedRoofingDataProcessor(config)
print('✅ Data processor initialized successfully')
"

# Setup validation script
print_step "Creating validation script..."
cat > scripts/validate_setup.sh << 'EOF'
#!/bin/bash
# Validation script for Pro Roofing AI setup

echo "🔍 Validating Pro Roofing AI setup..."

# Check virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
else
    echo "❌ Virtual environment not active"
    exit 1
fi

# Check required files
required_files=(".env" "config/training_config.yaml" "config/model_config.yaml")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ Required file exists: $file"
    else
        echo "❌ Missing required file: $file"
        exit 1
    fi
done

# Check data directory
if [ -d "data/raw" ] && [ "$(ls -A data/raw)" ]; then
    echo "✅ Raw data directory contains files"
else
    echo "⚠️  Raw data directory is empty"
fi

# Check Python imports
python -c "
try:
    from src.data.enhanced_data_processor import EnhancedRoofingDataProcessor
    from src.fine_tuning.trainer import RoofingAITrainer
    print('✅ All core modules import successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo "🎉 Setup validation completed successfully!"
EOF

chmod +x scripts/validate_setup.sh

print_step "Creating quick start script..."
cat > quick_start.sh << 'EOF'
#!/bin/bash
# Quick Start Script for Pro Roofing AI

echo "🚀 Pro Roofing AI Quick Start"
echo "============================="

# Run setup if not already done
if [ ! -d "venv" ] || [ ! -f ".env" ]; then
    echo "🔧 Running initial setup..."
    ./setup.sh
fi

# Activate environment
source venv/bin/activate

# Check if data needs processing
if [ ! -f "data/processed/ultimate_roofing_training.jsonl" ]; then
    echo "📊 Processing training data..."
    python src/data/enhanced_data_processor.py
fi

# Validate setup
echo "🔍 Validating setup..."
./scripts/validate_setup.sh

echo "✅ Pro Roofing AI is ready!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Review config/training_config.yaml"
echo "3. Run: python src/fine_tuning/trainer.py --config config/training_config.yaml"
echo ""
echo "For help: python src/fine_tuning/trainer.py --help"
EOF

chmod +x quick_start.sh

# Final setup verification
print_step "Final verification..."
if python -c "import src; print('✅ Package imports successful')" 2>/dev/null; then
    print_status "Package structure validated"
else
    print_error "Package import failed - please check installation"
    exit 1
fi

# Display completion message
echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo "1. Activate the virtual environment: ${YELLOW}source venv/bin/activate${NC}"
echo "2. Edit .env file with your API keys and configuration"
echo "3. Review and customize config/training_config.yaml"
echo "4. Process data: ${YELLOW}python src/data/enhanced_data_processor.py${NC}"
echo "5. Start training: ${YELLOW}python src/fine_tuning/trainer.py${NC}"
echo ""
echo -e "${BLUE}Useful Commands:${NC}"
echo "• Validate setup: ${YELLOW}./scripts/validate_setup.sh${NC}"
echo "• Quick start: ${YELLOW}./quick_start.sh${NC}"
echo "• Run tests: ${YELLOW}python -m pytest tests/${NC}"
echo ""
echo -e "${GREEN}Happy roofing! 🏠${NC}"