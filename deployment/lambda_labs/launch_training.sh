#!/bin/bash

# Pro Roofing AI - Lambda Labs Training Launcher
# One-command training start for Lambda Labs

set -e

echo "🎯 Pro Roofing AI - Training Launcher"
echo "=================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}📍 $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -f "src/fine_tuning/trainer.py" ]; then
    print_error "Please run this script from the pro-roofing-ai root directory"
    exit 1
fi

# Activate virtual environment
print_step "Activating virtual environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
    print_status "Virtual environment activated"
else
    print_error "Virtual environment not found. Run setup_lambda.sh first."
    exit 1
fi

# Check GPU availability
print_step "Checking GPU status..."
if nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1)
    print_status "GPU Status: $GPU_INFO"
else
    print_error "No GPU detected!"
    exit 1
fi

# Check environment file
print_step "Checking configuration..."
if [ ! -f ".env" ]; then
    print_error "Environment file (.env) not found. Please create it from .env.example"
    exit 1
fi

# Check if data needs processing
print_step "Checking training data..."
if [ ! -f "data/processed/ultimate_roofing_training.jsonl" ]; then
    print_status "Processing training data (this may take a while)..."
    python src/data/enhanced_data_processor.py
    
    if [ $? -eq 0 ]; then
        print_status "Data processing completed successfully"
    else
        print_error "Data processing failed"
        exit 1
    fi
else
    print_status "Training data already processed"
fi

# Display training configuration
print_step "Training Configuration Summary:"
echo "• Model: Qwen3-14B-Instruct"
echo "• LoRA fine-tuning with 4-bit quantization"
echo "• Training data: 400K+ roofing industry examples"
echo "• Expected duration: 6-10 hours (depending on GPU)"
echo "• Checkpoints saved every 500 steps"
echo "• WandB logging enabled"

# Ask for confirmation
echo ""
read -p "🚀 Ready to start training? This will take several hours. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Training cancelled"
    exit 0
fi

# Create training session
TRAINING_SESSION="roofing_ai_training_$(date +%Y%m%d_%H%M%S)"
print_step "Starting training session: $TRAINING_SESSION"

# Create screen session for training
screen -dmS "$TRAINING_SESSION" bash -c "
    source venv/bin/activate
    echo 'Starting Pro Roofing AI training...'
    echo 'Session: $TRAINING_SESSION'
    echo 'Started at: $(date)'
    echo ''
    
    # Run training with logging
    python src/fine_tuning/trainer.py --config config/training_config.yaml 2>&1 | tee logs/training/training_${TRAINING_SESSION}.log
    
    echo ''
    echo 'Training completed at: $(date)'
    echo 'Check logs/training/training_${TRAINING_SESSION}.log for details'
    
    # Keep session open
    exec bash
"

print_status "Training started successfully!"
echo ""
echo -e "${GREEN}Training Information:${NC}"
echo "• Session name: $TRAINING_SESSION"
echo "• Log file: logs/training/training_${TRAINING_SESSION}.log"
echo ""
echo -e "${YELLOW}Monitoring Commands:${NC}"
echo "• Attach to session: ${BLUE}screen -r $TRAINING_SESSION${NC}"
echo "• Detach from session: ${BLUE}Ctrl+A, then D${NC}"
echo "• View live logs: ${BLUE}tail -f logs/training/training_${TRAINING_SESSION}.log${NC}"
echo "• GPU monitoring: ${BLUE}nvtop${NC}"
echo "• List sessions: ${BLUE}screen -ls${NC}"
echo ""
echo -e "${GREEN}Training Progress:${NC}"
echo "• Check WandB dashboard for real-time metrics"
echo "• Model checkpoints saved in: models/checkpoints/"
echo "• Final model will be saved in: models/final/"
echo ""
print_status "Training is running in the background. Use 'screen -r $TRAINING_SESSION' to monitor."

# Optional: show initial logs
sleep 2
echo ""
echo -e "${BLUE}Initial training logs:${NC}"
tail -n 20 logs/training/training_${TRAINING_SESSION}.log 2>/dev/null || echo "Logs not yet available..."

echo ""
print_status "Training launched successfully! 🎉"