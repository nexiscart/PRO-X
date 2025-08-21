# 🏠 Pro Roofing AI - Lambda Labs Training Guide

## 🎯 Complete Setup for Lambda Labs GPU Training

Your **Pro Roofing AI** system is now ready for **Qwen3-14B fine-tuning** on **400K+ roofing industry examples**!

---

## 🚀 **Quick Start on Lambda Labs**

### **Option 1: One-Command Setup**
```bash
# SSH to your Lambda Labs instance
ssh ubuntu@your-lambda-instance-ip

# Clone and setup everything automatically
git clone https://github.com/nexiscart/PRO-X.git
cd PRO-X
./quick_start.sh

# Edit API keys (required)
nano .env

# Start training
python src/fine_tuning/trainer.py
```

### **Option 2: Lambda Labs Optimized**
```bash
# SSH to Lambda Labs instance
ssh ubuntu@your-lambda-instance-ip

# Clone repository
git clone https://github.com/nexiscart/PRO-X.git
cd PRO-X

# Run Lambda Labs specific setup
bash deployment/lambda_labs/setup_lambda.sh

# Edit your API keys
nano .env

# Launch training with monitoring
bash deployment/lambda_labs/launch_training.sh
```

---

## 📋 **What's Included**

### **🧠 Model Configuration**
- **Base Model**: Qwen3-14B-Instruct (32K context length)
- **Fine-tuning**: LoRA with 4-bit quantization
- **Memory Requirement**: ~48GB GPU memory (A100 recommended)
- **Training Time**: 6-10 hours on A100

### **📊 Training Data (400K+ Examples)**
✅ **12 Specialized JSON Datasets:**
- Blueprint Reading & Bidding (10K)
- Building Codes & Standards (8K)
- Commercial Construction Contracts (15K)
- Commercial Suppliers Mastery (15K)
- Roofing Expert Training (1,050)
- Roofing Mastery Training (70K)
- Roofing Systems Specifications (7K)
- Ultimate Roofing AI Master (200K)
- Ultimate Roofing Business Mastery (50K)
- Ultimate Roofing Mastery 70K Complete (70K)
- Ultimate Roofing Mastery 130K Complete (130K)
- World Class Sales Mastery (10K)

✅ **NRCA Manuals Processing:**
- PDF extraction and conversation generation
- Technical specifications and standards
- Installation procedures and best practices
- Safety protocols and code compliance

---

## 🔧 **Required Setup**

### **1. API Keys (Required)**
Edit `.env` file with your keys:
```env
OPENAI_API_KEY=your_openai_key_here
WANDB_API_KEY=your_wandb_key_here
HUGGINGFACE_TOKEN=your_hf_token_here
```

### **2. Lambda Labs Instance Requirements**
- **Recommended**: A100 (80GB) or A6000 (48GB)
- **Minimum**: RTX 6000 Ada (48GB)
- **OS**: Ubuntu 20.04/22.04
- **Storage**: 200GB+ available

---

## 📈 **Training Process**

### **Automatic Data Processing**
The system automatically:
1. **Copies datasets** from AI Drive (if available)
2. **Processes 400K+ examples** into conversation format
3. **Extracts PDF manuals** into training data
4. **Validates and cleans** all data
5. **Creates training/validation splits**

### **Training Configuration**
- **Batch Size**: 4 per device
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-4 with cosine scheduling
- **Epochs**: 3
- **Checkpoints**: Every 500 steps
- **Validation**: Every 100 steps

### **Monitoring**
- **WandB Integration**: Real-time metrics and loss curves
- **Local Logs**: `logs/training/trainer.log`
- **GPU Monitoring**: `nvtop` or `nvidia-smi`
- **Checkpoints**: Saved in `models/checkpoints/`

---

## 🎛️ **Monitoring Commands**

```bash
# View training progress
tail -f logs/training/trainer.log

# Monitor GPU usage
nvtop

# Check training session (if using screen)
screen -ls
screen -r roofing_ai_training_*

# View WandB dashboard
# https://wandb.ai/your-username/pro-roofing-ai
```

---

## 📁 **Directory Structure**

```
PRO-X/
├── config/                     # Training configurations
│   ├── training_config.yaml    # Qwen3-14B training setup
│   ├── model_config.yaml       # Model serving config
│   └── agent_config.yaml       # AI agents config
├── data/
│   ├── raw/                    # 400K+ JSON datasets
│   ├── processed/              # Processed training data
│   └── nrca_manuals/          # PDF manuals
├── src/
│   ├── data/                   # Data processing modules
│   ├── fine_tuning/           # Training pipeline
│   └── agents/                # AI agent system
├── models/
│   ├── checkpoints/           # Training checkpoints
│   └── final/                 # Final trained model
├── deployment/lambda_labs/    # Lambda Labs scripts
├── quick_start.sh            # One-command setup
└── README.md                 # Documentation
```

---

## ⚡ **Expected Results**

### **Training Metrics**
- **Training Loss**: Should decrease to ~0.5-1.0
- **Validation Loss**: Should track training loss
- **Perplexity**: Should decrease consistently
- **GPU Utilization**: 90%+ during training

### **Final Model**
- **Location**: `models/final/`
- **Format**: HuggingFace compatible
- **Size**: ~28GB (merged LoRA weights)
- **Capabilities**: Expert roofing knowledge + business intelligence

---

## 🔍 **Troubleshooting**

### **Common Issues**
1. **Out of Memory**: Reduce batch size in `config/training_config.yaml`
2. **CUDA Errors**: Check GPU compatibility and drivers
3. **API Rate Limits**: Ensure valid API keys in `.env`
4. **Data Missing**: Verify AI Drive datasets copied correctly

### **Support Commands**
```bash
# Check system resources
df -h                    # Disk space
free -h                  # Memory usage
nvidia-smi              # GPU status

# Validate installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "from src.data.enhanced_data_processor import *"
```

---

## 🎉 **Success Indicators**

✅ **Setup Complete When:**
- Virtual environment activated
- All dependencies installed
- 400K+ examples processed
- Training starts without errors
- WandB logging active
- GPU utilization high

✅ **Training Complete When:**
- Final model saved to `models/final/`
- Validation loss stabilized
- All epochs completed
- No critical errors in logs

---

## 📞 **Next Steps After Training**

1. **Test the Model**: Use inference scripts to test responses
2. **Deploy Agents**: Configure and deploy the AI agent system
3. **Integration**: Connect to CRM and email systems
4. **Monitoring**: Set up production monitoring

---

**🚀 Your Pro Roofing AI system is ready for training on Lambda Labs!**

**Repository**: https://github.com/nexiscart/PRO-X

Just clone, run the setup script, add your API keys, and start training! 🏠✨