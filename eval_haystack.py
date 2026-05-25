import time
import os
import sys
import torch
from peft import PeftModel
import config
import model_manager

def generate_haystack_context():
    """Generates ~1000-1500 tokens of noisy log entries with a hidden anomaly (the needle)."""
    logs = []
    # Generate 50 log entries to construct a standard context haystack
    for i in range(50):
        timestamp = f"May 24 18:{i//60:02d}:{i%60:02d}"
        if i == 27:
            # The Needle: unauthorized ssh login
            logs.append(f"{timestamp} server sshd[22415]: Accepted publickey for anomalous_hacker from 192.168.99.199 port 61205 ssh2: RSA SHA256:attackersignaturehere")
        elif i == 38:
            # Clue 2: cron privilege escalation
            logs.append(f"{timestamp} server CRON[22501]: pam_unix(cron:session): session opened for user anomalous_hacker by (uid=0)")
        else:
            # Background noise (SSH logouts, Nginx requests, systemd actions)
            if i % 3 == 0:
                logs.append(f"{timestamp} server sshd[{1000+i}]: Connection closed by authenticating user guest 192.168.1.{i} port {4000+i} [preauth]")
            elif i % 3 == 1:
                logs.append(f"{timestamp} server Nginx: 192.168.2.{i} - - [{timestamp}] \"GET /api/v1/health HTTP/1.1\" 200 45 \"-\" \"curl/7.81.0\"")
            else:
                logs.append(f"{timestamp} server systemd[1]: Starting Daily apt upgrade and clean activities...")
    
    return "\n".join(logs)

def evaluate_model(model, tokenizer, use_lora_label):
    """Runs A/B diagnostics on the model context retrieval capabilities."""
    task = "Identify the username and IP address of the unauthorized shell logon inside the logs."
    context = generate_haystack_context()
    previous_action = "Initial log query."

    # Format the prompt with our system monologue instructions
    messages = [
        {
            "role": "system",
            "content": (
                "You are an agentic AI system. You possess advanced cognitive heuristics to scan messy server contexts "
                "and previous steps. Always explain your detailed internal thought process (Thought Process) showing "
                "how and why you are searching specific parts of the context, then output your final Next Action."
            )
        },
        {
            "role": "user",
            "content": f"Task: {task}\n\nContext:\n{context}\n\nPrevious Action: {previous_action}"
        }
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

    print(f"\nEvaluating [{use_lora_label}] model on Haystack Needle-Search...")

    # 1. Measure Time to First Token (TTFT)
    start_ttft = time.time()
    with torch.no_grad():
        tokenizer.decode(
            model.generate(
                **inputs,
                max_new_tokens=1,
                pad_token_id=tokenizer.eos_token_id
            )[0]
        )
    ttft_duration = time.time() - start_ttft

    # 2. Measure complete generation duration and compile response
    start_full = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    total_duration = time.time() - start_full
    
    # Extract only new generated tokens
    generated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    
    print(f"[{use_lora_label}] Response completed in {total_duration:.2f}s.")
    print("\n--- Response Sample ---")
    print(generated_text)
    print("------------------------\n")

    # Evaluate metrics
    has_thought = "Thought Process:" in generated_text
    has_action = "Next Action:" in generated_text
    found_username = "anomalous_hacker" in generated_text
    found_ip = "192.168.99.199" in generated_text
    accuracy = found_username and found_ip

    return {
        "label": use_lora_label,
        "ttft": ttft_duration,
        "total_time": total_duration,
        "has_thought": has_thought,
        "has_action": has_action,
        "accuracy": accuracy,
        "response": generated_text
    }

def run_ab_test():
    """Loads and benchmarks baseline versus MLP-LoRA models, outputting a comparative table."""
    print("=== STARTING A/B BENCHMARK HAYSTACK TEST ===")

    # 1. Evaluate Baseline Model
    model, tokenizer = model_manager.load_base_model()
    baseline_metrics = evaluate_model(model, tokenizer, "Baseline 4-Bit")
    
    # Clean baseline resources
    print("Unloading Baseline Model...")
    del model
    model_manager.clear_gpu_memory()

    # 2. Evaluate Fine-tuned Heuristic Model
    if not os.path.exists(os.path.join(config.OUTPUT_DIR, "adapter_config.json")):
        print(f"\n[WARNING] Trained LoRA adapter not found at {config.OUTPUT_DIR}!")
        print("Please run agent_pipeline.py first to execute the sleep cycle training.")
        heuristic_metrics = {
            "label": "Heuristic (Trained LoRA)",
            "ttft": 0.0,
            "total_time": 0.0,
            "has_thought": False,
            "has_action": False,
            "accuracy": False,
            "response": "N/A - Run training first."
        }
    else:
        # Load base model, then attach saved LoRA weights
        base_model, val_tokenizer = model_manager.load_base_model()
        peft_model = PeftModel.from_pretrained(base_model, config.OUTPUT_DIR)
        peft_model.eval()
        
        heuristic_metrics = evaluate_model(peft_model, val_tokenizer, "Heuristic MLP-LoRA")
        
        # Clean PEFT resources
        print("Unloading Heuristic Model...")
        del peft_model
        del base_model
        model_manager.clear_gpu_memory()

    # 3. Print Comparison Table
    print("\n" + "="*60)
    print("                 BENCHMARK PERFORMANCE SUMMARY")
    print("="*60)
    
    table_header = f"| {'Metric':<25} | {'Baseline (No LoRA)':<20} | {'Heuristic (MLP-LoRA)':<20} |"
    divider = f"|{'-'*27}|{'-'*22}|{'-'*22}|"
    
    acc_baseline = "PASS (100%)" if baseline_metrics["accuracy"] else "FAIL (0%)"
    acc_heuristic = "PASS (100%)" if heuristic_metrics["accuracy"] else "FAIL (0%)"
    
    thought_baseline = "Yes" if baseline_metrics["has_thought"] else "No"
    thought_heuristic = "Yes" if heuristic_metrics["has_thought"] else "No"
    
    action_baseline = "Yes" if baseline_metrics["has_action"] else "No"
    action_heuristic = "Yes" if heuristic_metrics["has_action"] else "No"

    print(table_header)
    print(divider)
    print(f"| {'Search Success (Accuracy)':<25} | {acc_baseline:<20} | {acc_heuristic:<20} |")
    print(f"| {'Time to First Token (TTFT)':<25} | {baseline_metrics['ttft']:<18.4f}s | {heuristic_metrics['ttft']:<18.4f}s |")
    print(f"| {'Total Generation Time':<25} | {baseline_metrics['total_time']:<18.4f}s | {heuristic_metrics['total_time']:<18.4f}s |")
    print(f"| {'Generates Thought Monologue':<25} | {thought_baseline:<20} | {thought_heuristic:<20} |")
    print(f"| {'Generates Action Field':<25} | {action_baseline:<20} | {action_heuristic:<20} |")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_ab_test()
