# 🏠 Pro Roofing AI - Advanced AI System for Roofing Industry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI GPT](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)](https://openai.com/)

## 🎯 Overview

Pro Roofing AI is a comprehensive artificial intelligence system designed specifically for the roofing industry. It combines fine-tuned language models with intelligent agent systems to automate and optimize various aspects of roofing business operations.

### 🚀 Key Features

- **🧠 Qwen3-14B Fine-Tuned Model**: Advanced model trained on 400K+ roofing industry examples + NRCA manuals
- **🤖 Intelligent Agent System**: Multi-agent orchestration for lead generation, bidding, and customer management
- **📊 Advanced Data Processing**: Handles JSON datasets + PDF manual extraction and processing
- **💰 Automated Bidding**: AI-powered bidding system with blueprint reading capabilities
- **📧 Email Automation**: Intelligent email outreach and customer communication
- **🔗 CRM Integration**: Seamless integration with popular CRM systems
- **📈 Performance Analytics**: Real-time monitoring and optimization
- **⚡ Lambda Labs Optimized**: One-command setup and training for Lambda Labs GPU instances

## 📁 Project Structure

```
pro-roofing-ai/
├── 📄 README.md                              # Main documentation
├── 📄 LICENSE                                # MIT License
├── 📄 .gitignore                             # Git ignore rules
├── 📄 .env.example                           # Environment template
├── 📄 requirements.txt                       # Python dependencies
├── 🚀 setup.sh                               # Environment setup script
├── 🚀 quick_start.sh                         # One-command launcher
├── 📁 config/                                # Configuration files
├── 📁 data/                                  # Data directory with 12 specialized datasets
├── 📁 src/                                   # Source code
│   ├── 📁 fine_tuning/                       # Training pipeline
│   ├── 📁 data/                              # Data processing
│   └── 📁 agents/                            # AI Agent system
├── 📁 models/                                # Model storage
├── 📁 logs/                                  # Logging
├── 📁 scripts/                               # Utility scripts
├── 📁 tests/                                 # Testing suite
├── 📁 notebooks/                             # Jupyter notebooks
├── 📁 deployment/                            # Deployment configs
├── 📁 docs/                                  # Documentation
└── 📁 workflows/                             # Automation workflows
```

## 🛠️ Installation

### Quick Start (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd pro-roofing-ai

# Run the quick setup
chmod +x quick_start.sh
./quick_start.sh
```

### Manual Installation

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Run setup script
chmod +x setup.sh
./setup.sh
```

## 🔧 Configuration

Create your `.env` file from the template:

```bash
cp .env.example .env
```

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `LAMBDA_API_KEY`: Lambda Labs API key (for training)
- `DATABASE_URL`: Database connection string
- `SMTP_CONFIG`: Email server configuration

## 📊 Training Data (400K+ Examples)

The system includes 12 specialized JSON datasets + NRCA manuals:

### JSON Datasets:
1. **Blueprint Reading & Bidding** (10K examples)
2. **Building Codes & Standards** (8K examples)
3. **Commercial Construction Contracts** (15K examples)
4. **Commercial Suppliers Mastery** (15K examples)
5. **Roofing Expert Training** (1,050 examples)
6. **Roofing Mastery Training** (70K examples)
7. **Roofing Systems Specifications** (7K examples)
8. **Ultimate Roofing AI Master** (200K examples)
9. **Ultimate Roofing Business Mastery** (50K examples)
10. **Ultimate Roofing Mastery 70K Complete** (70K examples)
11. **Ultimate Roofing Mastery 130K Complete** (130K examples)
12. **World Class Sales Mastery** (10K examples)

### NRCA Manuals:
- Technical specifications and standards
- Installation procedures and best practices
- Safety protocols and code compliance
- Material performance characteristics

## 🚀 Quick Start

### One-Command Setup & Training

```bash
# Complete setup and data processing
./quick_start.sh

# Then start training
python src/fine_tuning/trainer.py --config config/training_config.yaml
```

### Lambda Labs GPU Training

```bash
# On Lambda Labs instance, run:
bash deployment/lambda_labs/setup_lambda.sh
bash deployment/lambda_labs/launch_training.sh
```

### Manual Training Steps

```bash
# 1. Process and validate data (includes AI Drive datasets + NRCA manuals)
python src/data/enhanced_data_processor.py

# 2. Start Qwen3-14B fine-tuning
python src/fine_tuning/trainer.py --config config/training_config.yaml
```

### Running Agents

```bash
# Start the agent orchestrator
python src/agents/orchestrator.py

# Or run individual agents
python src/agents/lead_agent.py
python src/agents/bidding_agent.py
```

### Monitoring

```bash
# Monitor training progress
./scripts/monitor_training.sh

# View performance dashboard
./scripts/performance_dashboard.sh
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test suites
python -m pytest tests/test_trainer.py
python -m pytest tests/test_agents.py
```

## 📈 Performance Optimization

- **GPU Training**: Optimized for Lambda Labs and cloud GPU instances
- **Memory Management**: Efficient data loading and model checkpointing
- **Distributed Training**: Support for multi-GPU training
- **Model Quantization**: Reduced model size for deployment

## 🔗 Integrations

### CRM Systems
- HubSpot
- Salesforce
- Pipedrive
- Custom CRM APIs

### Email Platforms
- Gmail API
- Outlook API
- SendGrid
- Mailgun

### Workflow Automation
- n8n workflows
- CrewAI integration
- Zapier connections

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Training Guide](docs/training.md)
- [Agent Documentation](docs/agents.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📧 Email: support@proroofingai.com
- 💬 Discord: [Join our community](https://discord.gg/proroofingai)
- 📖 Documentation: [docs.proroofingai.com](https://docs.proroofingai.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)

## 🙏 Acknowledgments

- OpenAI for GPT models
- Lambda Labs for GPU infrastructure
- The roofing industry professionals who provided expertise
- Open source community for tools and libraries

---

**Built with ❤️ for the Roofing Industry**