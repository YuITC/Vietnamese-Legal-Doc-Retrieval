import os
import torch
import random
import numpy as np


# ===== Data settings =====
os.makedirs('data', exist_ok=True)
os.makedirs('data/original', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/retrieval', exist_ok=True)


# ===== Model settings =====
MODEL_ID   = 'google-bert/bert-base-multilingual-cased'
MODEL_NAME = 'VN-legalDocs-SBERT'

CACHE_DIR  = f"cache/{MODEL_NAME}"
OUTPUT_DIR = f"models/{MODEL_NAME}"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===== Reproducibility =====
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Reproducibility: deterministic=True, benchmark=False
# Optimize inference/training speed: deterministic=False, benchmark=True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True


# ===== Hyperparameters =====
MAX_SEQ_LEN = 512
EPOCHS      = 5
LR          = 3e-5
BATCH_SIZE  = 128
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")