from trl import SFTTrainer, SFTConfig
import config

def run_sleep_cycle(model, tokenizer, dataset):
    """
    Runs the SFT 'sleep cycle' training loop using PEFT/QLoRA on the compiled heuristics dataset.
    Optimized to fit robustly inside the 11GB VRAM RTX 2080 Ti ceiling.
    """
    print(f"\n--- [MicroTrainer] Entering Sleep Cycle (Training Session) ---")
    print(f"Dataset contains {len(dataset)} cognitive traces.")
    
    # SFT configurations optimized specifically for single-GPU RTX 2080 Ti execution
    training_args = SFTConfig(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.MICRO_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.NUM_EPOCHS,
        optim="paged_adamw_8bit",           # Crucial optimization: offloads states to massive host CPU RAM
        gradient_checkpointing=True,        # Minimizes peak activation VRAM usage significantly
        fp16=False,                         # Disable GradScaler which crashes on bfloat16 parameters
        bf16=True,                          # Enable stable native bfloat16 training (supported on modern PyTorch/CUDA)
        logging_steps=1,
        save_strategy="no",                 # Prevent saving intermediate checkpoints (saves VRAM/disk)
        report_to="none",                   # Disable wandb/tensorboard overhead
        remove_unused_columns=False,
        max_length=config.MAX_SEQ_LENGTH,   # Correct parameter for newer TRL SFTConfig
    )
    
    # Initialize SFTTrainer. It automatically parses ChatML structure (messages list) and tokenizes it.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args
    )
    
    print("Executing training steps...")
    trainer.train()
    
    print(f"Training completed. Saving trained LoRA adapter to: {config.OUTPUT_DIR}")
    trainer.model.save_pretrained(config.OUTPUT_DIR)
    print("--- [MicroTrainer] Sleep Cycle Complete ---\n")

