import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

LARGE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # Double the size of our 3B model
LOGS_DIR = "./data/logs"

def load_concatenated_logs():
    """Loads all generated logs concatenated in a single string (~15,000 tokens)."""
    concatenated = []
    files = ["nginx.log", "syslog", "database.log", "auth.log", "cron.log"]
    for fn in files:
        path = os.path.join(LOGS_DIR, fn)
        if not os.path.exists(path):
            print(f"[Error] Log file '{fn}' does not exist. Run agent_extreme_context.py first!")
            sys.exit(1)
        with open(path, "r") as f:
            concatenated.append(f"=== FILE: {fn} ===\n" + f.read())
    return "\n\n".join(concatenated)

def run_scenario_c():
    """Scenario C: Powerful 7B Model on CPU. Tries to brute-force ingest the entire 15,000-token prompt."""
    print("\n" + "="*60)
    print("      SCENARIO C: POWERFUL 7B MODEL CPU BRUTE-FORCE INGESTION")
    print("="*60)
    
    print(f"Loading large tokenizer for {LARGE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(LARGE_MODEL_ID, trust_remote_code=True)
    
    print(f"Loading large model {LARGE_MODEL_ID} fully in host CPU RAM (64GB available)...")
    # Loaded in float16 to fit in memory (takes ~14GB RAM)
    start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        LARGE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},  # Direct loading on host system CPU RAM
        trust_remote_code=True
    )
    print(f"Model loaded successfully in {time.time() - start_load:.2f}s.")
    
    task = "Identify the entry vector of the compromise and the subsequent privilege escalation chain."
    print("Loading 15,000-token log dump...")
    full_context = load_concatenated_logs()
    
    messages = [
        {
            "role": "system",
            "content": "You are a system diagnostician agent. Analyze the provided logs and identify the entry vector and privilege escalation chain."
        },
        {
            "role": "user",
            "content": f"Task: {task}\n\nLogs Context:\n{full_context}"
        }
    ]
    
    print("Tokenizing prompt...")
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    print(f"Prompt sequence length: {inputs['input_ids'].shape[1]} tokens.")
    
    print("Starting generation on CPU (execution will be slow, taking a few minutes)...")
    start_time = time.time()
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id
            )
        duration = time.time() - start_time
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print(f"Scenario C completed successfully in {duration:.2f}s.")
        print("\n--- Model Response ---")
        print(response)
        print("----------------------\n")
        
        # Check metrics
        found_sqli = "nginx.log" in response.lower() or "sql injection" in response.lower() or "union select" in response.lower()
        found_priv = "auth.log" in response.lower() or "sudo" in response.lower() or "privilege escalation" in response.lower()
        accuracy = found_sqli and found_priv
        
        print("\n" + "="*60)
        print("                 BENCHMARK SCENARIO C SUMMARY")
        print("="*60)
        print(f"Accuracy (Found compromise path): {'PASS' if accuracy else 'FAIL'}")
        print(f"Total Computation Duration: {duration:.2f}s")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n[FAILURE] Scenario C failed during CPU execution: {str(e)}")

if __name__ == "__main__":
    run_scenario_c()
