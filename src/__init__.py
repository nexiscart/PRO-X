"""Pro Roofing AI - Advanced AI System for Roofing Industry"""

__version__ = "1.0.0"
__author__ = "Pro Roofing AI Team"
__email__ = "support@proroofingai.com"
__description__ = "Comprehensive AI system for commercial roofing industry automation"

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VALIDATION_DATA_DIR = DATA_DIR / "validation"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
BASE_MODELS_DIR = MODELS_DIR / "base"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
FINAL_MODELS_DIR = MODELS_DIR / "final"

# Configuration directory
CONFIG_DIR = PROJECT_ROOT / "config"

# Logging directory
LOGS_DIR = PROJECT_ROOT / "logs"