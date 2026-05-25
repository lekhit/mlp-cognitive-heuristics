import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import config

def load_base_model():
    """Loads tokenizer and base model in 4-bit precision optimized for Turing (2080 Ti)."""
    print(f"Loading tokenizer for {config.MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Right padding for training stability

    print(f"Loading quantized base model {config.MODEL_ID} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16  # bfloat16 for native model compatibility
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},  # Enforce loading on single GPU 0
        trust_remote_code=True,
        torch_dtype=torch.bfloat16  # Native bfloat16 loading to prevent scaling errors
    )

    # Disable KV cache during training to allow gradient checkpointing
    model.config.use_cache = False

    return model, tokenizer

def attach_mlp_lora(model):
    """Prepares 4-bit model and attaches MLP-targeted LoRA adapter."""
    print("Preparing model for kbit training...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print(f"Attaching MLP LoRA adapter (Rank: {config.LORA_RANK}, Target Modules: {config.TARGET_MODULES})...")
    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def clear_gpu_memory():
    """Forcefully clear CPU/GPU garbage collector and cache to prevent OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("VRAM cleared successfully.")
