import os

# Central Hyperparameters & Configurations

# Model configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Sequence length for training and generation
MAX_SEQ_LENGTH = 1024

# LoRA adapter settings
# Targeting feed-forward network (MLP) layers specifically
# In Qwen2 and Llama, MLP layers are gate_proj, up_proj, and down_proj
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]

# Quantized training settings (VRAM optimized for 11GB RTX 2080 Ti)
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

# Data settings
BUFFER_THRESHOLD = 5  # Number of traces to buffer before triggering a sleep cycle (micro-training)
DATASET_PATH = "./data/heuristic_traces.jsonl"
OUTPUT_DIR = "./outputs/heuristic_lora"

# Ensure data and output directories exist
os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
