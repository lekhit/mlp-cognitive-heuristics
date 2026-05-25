import os
import sys
import time
import torch
from peft import PeftModel
import config
import model_manager

# Directory to store our massive log files
LOGS_DIR = "./data/logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def generate_large_logs_database():
    """Programmatically generates a 10,000-15,000-token multi-file server log database."""
    print("[ExtremeContext] Generating log files...")
    
    # 1. Generate nginx.log (~2,500 tokens) with SQL Injection needle
    nginx_path = os.path.join(LOGS_DIR, "nginx.log")
    with open(nginx_path, "w") as f:
        for i in range(30):
            timestamp = f"24/May/2026:19:{i//60:02d}:{i%60:02d} +0000"
            if i == 15:
                # The Needle: SQL Injection exploit attempt
                f.write(f'192.168.1.99 - - [{timestamp}] "POST /api/v1/auth/login?user_id=1%20UNION%20SELECT%20username,%20password%20FROM%20users HTTP/1.1" 200 1024 "http://vuln-site.com" "Mozilla/5.0"\n')
            else:
                # Noise
                f.write(f'192.168.1.{10+i%20} - - [{timestamp}] "GET /static/assets/logo.png HTTP/1.1" 200 4502 "http://vuln-site.com" "Mozilla/5.0"\n')
                f.write(f'192.168.1.{30+i%10} - - [{timestamp}] "GET /api/v1/health HTTP/1.1" 200 42 "http://vuln-site.com" "curl/7.81.0"\n')
                
    # 2. Generate syslog (~2,000 tokens) of standard noise
    syslog_path = os.path.join(LOGS_DIR, "syslog")
    with open(syslog_path, "w") as f:
        for i in range(30):
            timestamp = f"May 24 19:{i//60:02d}:{i%60:02d}"
            f.write(f"{timestamp} host kernel: [ {1000.0+i:.4f}] usb 1-1: new high-speed USB device number {i%10+2} using xhci_hcd\n")
            f.write(f"{timestamp} host systemd[1]: Activated service daemon-manager-{i%5}.service successfully.\n")

    # 3. Generate database.log (~2,500 tokens) showing successful SQLi query execution
    db_path = os.path.join(LOGS_DIR, "database.log")
    with open(db_path, "w") as f:
        for i in range(30):
            timestamp = f"2026-05-24 19:{i//60:02d}:{i%60:02d}"
            if i == 18:
                # The Needle: DB executing the SQLi and returning credentials
                f.write(f"{timestamp} [DB_EXEC] [QUERY] SELECT username, password FROM users\n")
                f.write(f"{timestamp} [DB_EXEC] [RESULT] Row 1: admin, $2b$12$hashedpasswordvaluehere2026adminsecret\n")
            else:
                f.write(f"{timestamp} [DB_EXEC] [QUERY] SELECT * FROM sessions WHERE session_token = 'sess_token_{i}'\n")
                f.write(f"{timestamp} [DB_EXEC] [RESULT] Row 1: session_id={i}, user_id={100+i}\n")

    # 4. Generate auth.log (~2,000 tokens) with root privilege escalation needle
    auth_path = os.path.join(LOGS_DIR, "auth.log")
    with open(auth_path, "w") as f:
        for i in range(25):
            timestamp = f"May 24 19:{i//60:02d}:{i%60:02d}"
            if i == 12:
                # The Needle: successful login + passwordless sudo privilege escalation to root
                f.write(f"{timestamp} host sshd[33104]: Accepted password for admin from 192.168.1.99 port 41202 ssh2\n")
                f.write(f"{timestamp} host sudo:    admin : TTY=pts/2 ; PWD=/home/admin ; USER=root ; COMMAND=/bin/bash (Warning: passwordless sudo root privilege escalated!)\n")
            else:
                f.write(f"{timestamp} host sshd[{12000+i}]: Failed password for invalid user guest from 192.168.5.{i%10} port {35000+i} ssh2\n")
                f.write(f"{timestamp} host CRON[33200]: pam_unix(cron:session): session opened/closed for user root\n")

    # 5. Generate cron.log (~1,500 tokens) of standard cron job noise
    cron_path = os.path.join(LOGS_DIR, "cron.log")
    with open(cron_path, "w") as f:
        for i in range(20):
            timestamp = f"May 24 19:{i//60:02d}:{i%60:02d}"
            f.write(f"{timestamp} host CRON[44102]: (root) CMD (   /usr/local/bin/cleanup_temp_files.sh > /dev/null 2>&1)\n")
            f.write(f"{timestamp} host CRON[44105]: (www-data) CMD (   python3 /var/www/html/api/cron_sync.py)\n")

    print("[ExtremeContext] Log database successfully generated under ./data/logs/.")

def load_concatenated_logs():
    """Helper to load all logs concatenated in a single string (~15,000 tokens)."""
    concatenated = []
    files = ["nginx.log", "syslog", "database.log", "auth.log", "cron.log"]
    for fn in files:
        path = os.path.join(LOGS_DIR, fn)
        with open(path, "r") as f:
            concatenated.append(f"=== FILE: {fn} ===\n" + f.read())
    return "\n\n".join(concatenated)

def run_scenario_a():
    """Scenario A: Baseline 3B brute-force. Proves OOM or extreme latency/failure."""
    print("\n" + "="*60)
    print("      SCENARIO A: BASELINE 3B MODEL BRUTE-FORCE INGESTION")
    print("="*60)
    
    # Load raw model
    model, tokenizer = model_manager.load_base_model()
    
    task = "Identify the entry vector of the compromise and the subsequent privilege escalation chain."
    print("Concatenating all log files into a single context (~15,000 tokens)...")
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
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
    print(f"Prompt tensor shape: {inputs['input_ids'].shape} (Sequence length: {inputs['input_ids'].shape[1]} tokens)")
    
    print("Starting generation on GPU 0 (expecting potential Out-of-Memory)...")
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
        print(f"Scenario A successfully executed in {duration:.2f}s (No OOM).")
        print("\n--- Model Response ---")
        print(response)
        print("----------------------\n")
    except torch.cuda.OutOfMemoryError as e:
        print("\n[CRITICAL FAILURE] SCENARIO A HIT CUDA OUT OF MEMORY!")
        print("Detail: Prefill attention matrix allocation exceeded GPU VRAM capacity.")
        print(str(e))
    except Exception as e:
        print(f"\n[FAILURE] Scenario A failed with exception: {str(e)}")
        
    # Unload
    del model
    model_manager.clear_gpu_memory()

def run_scenario_b():
    """Scenario B: Heuristic-Guided 3B Agent. Uses fine-tuned LoRA & dynamic dynamic search."""
    print("\n" + "="*60)
    print("      SCENARIO B: HEURISTIC-GUIDED 3B AGENT (MLP-LoRA)")
    print("="*60)
    
    # Load base model + trained MLP-LoRA adapter
    base_model, tokenizer = model_manager.load_base_model()
    model = PeftModel.from_pretrained(base_model, config.OUTPUT_DIR)
    model.eval()
    
    task = "Identify the entry vector of the compromise and the subsequent privilege escalation chain."
    
    # Directory index containing a brief listing of available logs
    directory_index = (
        "Available Log Files in Database Directory:\n"
        "- nginx.log (Web request records)\n"
        "- syslog (Daemon and cluster system logs)\n"
        "- database.log (Database execution and result records)\n"
        "- auth.log (User authentications and shell access status)\n"
        "- cron.log (Recurring cluster scheduler logs)"
    )
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an agentic AI system. You possess advanced cognitive heuristics to scan messy server contexts "
                "and previous steps. Always explain your detailed internal thought process (Thought Process) showing "
                "how and why you are searching specific parts of the context, then output your final Next Action in the exact format:\n"
                "Thought Process: <reasoning>\n"
                "Next Action: read <file_name>\n"
                "When you have identified the full entry vector and privilege escalation chain, conclude with:\n"
                "Next Action: complete"
            )
        },
        {
            "role": "user",
            "content": f"Task: {task}\n\nDirectory Structure:\n{directory_index}\n\nPrevious Action: Initial Directory Status Query."
        }
    ]
    
    history_log = []
    current_action = "Initial Directory Query"
    
    start_time = time.time()
    step_count = 0
    max_steps = 6
    success = False
    
    while step_count < max_steps:
        step_count += 1
        print(f"\n--- [Heuristic Agent] Step {step_count} ---")
        
        # Tokenize and generate next planning block
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
        print(f"Current step input size: {inputs['input_ids'].shape[1]} tokens (Very small VRAM foot-print!)")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print("\nGenerated Agent Monologue:")
        print(generated_text)
        
        # Parse output command
        lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
        thought_line = ""
        action_line = ""
        for line in lines:
            if line.startswith("Thought Process:"):
                thought_line = line
            elif line.startswith("Next Action:"):
                action_line = line
                
        # Fallback parser if tags were formatted slightly differently
        if not action_line:
            for line in lines:
                if "Next Action:" in line:
                    action_line = line[line.find("Next Action:"):]
                    
        # Log planning history
        history_log.append(generated_text)
        
        if "complete" in action_line.lower():
            print("\n[SUCCESS] Agent signaled task completion! Exploit chain fully mapped.")
            success = True
            break
            
        # Handle file read action
        if "read" in action_line.lower():
            # Extract file name
            parts = action_line.split(" ")
            file_name = parts[-1].strip().strip("'\"")
            
            # Read file context
            path = os.path.join(LOGS_DIR, file_name)
            if os.path.exists(path):
                print(f"\n[Environment] Intercepted Action. Loading content of '{file_name}'...")
                with open(path, "r") as f:
                    file_content = f.read()
                
                # We feed ONLY the requested file contents as context to save VRAM!
                # Update Chat history
                messages = [
                    messages[0], # Maintain system prompt
                    {
                        "role": "user",
                        "content": (
                            f"Task: {task}\n\n"
                            f"File Context for '{file_name}':\n{file_content}\n\n"
                            f"Previous Actions Log:\n" + "\n".join([f"Step {idx+1}: {act}" for idx, act in enumerate(history_log)])
                        )
                    }
                ]
            else:
                print(f"\n[Environment Error] File '{file_name}' not found.")
                messages.append({"role": "user", "content": f"Error: File '{file_name}' not found in logs directory."})
        else:
            print("\n[Agent Loop Warning] Next Action did not match standard format. Ending loop.")
            break
            
    total_duration = time.time() - start_time
    print(f"\nScenario B finished in {total_duration:.2f}s.")
    print(f"Total Steps Executed: {step_count}")
    print(f"Agent Task Outcome: {'SUCCESS' if success else 'FAILED'}")
    
    # Unload
    del model
    del base_model
    model_manager.clear_gpu_memory()

if __name__ == "__main__":
    generate_large_logs_database()
    run_scenario_a()
    run_scenario_b()
