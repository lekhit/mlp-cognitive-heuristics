import os
import sys
import torch
from peft import PeftModel
import config
import model_manager
from heuristic_builder import LogBuffer
import micro_trainer

def generate_synthetic_data(builder):
    """
    Populates the database with 10 high-quality 'cognitive search heuristic' traces.
    These traces teach the model how to scan logs, check previous actions, and plan steps.
    """
    print("[AgentPipeline] Generating high-quality baseline heuristic traces...")
    
    # 10 synthetic steps of an agent investigating a complex memory leak and network anomaly
    traces = [
        {
            "task": "Diagnose the memory leak in the microservices cluster.",
            "context": "System Logs:\n- service_auth: RSS=120MB, cpu=0.5%\n- service_gateway: RSS=250MB, cpu=1.2%\n- service_db: RSS=1.1GB, cpu=4.5%\n- service_analytics: RSS=4.2GB, cpu=88.5% (Warning: RAM leak suspected)",
            "prev_action": "Initial cluster status query.",
            "thought": "I need to diagnose a memory leak. Scanning the cluster status logs, service_analytics is consuming 4.2GB of RAM and 88.5% CPU. This is highly disproportionate compared to service_db (1.1GB) and other services. I will focus my search specifically on service_analytics logs to identify why memory is leaking.",
            "next_action": "inspect service_analytics logs"
        },
        {
            "task": "Diagnose the memory leak in the microservices cluster.",
            "context": "service_analytics Logs:\n- 19:40:02 [INFO] Worker pool initialized with 20 threads.\n- 19:41:15 [WARN] Buffer queue size exceeded 5000 items. Memory allocated: 1.8GB\n- 19:42:30 [WARN] Buffer queue size exceeded 20000 items. Memory allocated: 3.2GB\n- 19:43:55 [ERROR] OutOfMemory exception imminent. Queue size: 55000 items. Thread state: BLOCKED on KafkaWriter.",
            "prev_action": "inspect service_analytics logs",
            "thought": "The logs for service_analytics show buffer queues growing rapidly from 5000 items to 55000 items, while memory allocated scales from 1.8GB to 3.2GB before hitting an OOM exception. The threads are BLOCKED on 'KafkaWriter'. This indicates a bottleneck writing to Kafka, causing queue backpressure. I need to search the Kafka cluster status to verify its availability.",
            "next_action": "inspect kafka_cluster status"
        },
        {
            "task": "Diagnose the memory leak in the microservices cluster.",
            "context": "Kafka Cluster Logs:\n- Broker 1 (10.0.0.5): Online. In-Sync-Replicas: [1, 2]. Disk usage: 98% (Alert: Read-Only mode triggered)\n- Broker 2 (10.0.0.6): Online. In-Sync-Replicas: [1, 2].\n- client_id: service_analytics blocked on topic 'analytics_stream' due to Broker 1 disk space exception.",
            "prev_action": "inspect kafka_cluster status",
            "thought": "Kafka Broker 1 is in read-only mode because disk usage reached 98%. This disk alert is preventing service_analytics from writing to the 'analytics_stream' topic, causing the queue backup and memory leak in service_analytics. To resolve the memory leak, we must address the Kafka disk space. I should search for old topics or logs on Broker 1 that can be purged.",
            "next_action": "search broker_1 disk files"
        },
        {
            "task": "Diagnose the memory leak in the microservices cluster.",
            "context": "Broker 1 Disk Files:\n- /var/log/kafka/server.log.2025: 180GB (Old uncompressed logs)\n- /data/kafka/analytics_stream-0: 12GB\n- /data/kafka/auth_events-0: 2GB",
            "prev_action": "search broker_1 disk files",
            "thought": "Scanning the disk files on Broker 1, I locate an old uncompressed server log file `/var/log/kafka/server.log.2025` occupying 180GB of space. The active topic data (`analytics_stream-0`) is only 12GB. Purging the old 180GB log file is the safest way to reclaim space and restore Kafka to read-write status. I will delete this log file.",
            "next_action": "delete file /var/log/kafka/server.log.2025"
        },
        {
            "task": "Diagnose the memory leak in the microservices cluster.",
            "context": "System Output:\n- File /var/log/kafka/server.log.2025 deleted successfully.\n- Broker 1 Disk usage decreased to 14%. Broker 1 status changed from Read-Only to Read-Write.",
            "prev_action": "delete file /var/log/kafka/server.log.2025",
            "thought": "The old log file has been purged, reducing Broker 1 disk usage to 14% and restoring it to Read-Write. The backpressure should now relieve as service_analytics flushes its buffer. I should check the status of service_analytics to confirm its queue size and RAM footprint are declining.",
            "next_action": "inspect service_analytics status"
        },
        {
            "task": "Investigate SSH authentication anomalies in secure host logs.",
            "context": "Authentication Logs (/var/log/auth.log):\n- May 24 12:01:05 host sshd[12401]: Failed password for invalid user admin from 192.168.1.150 port 54102 ssh2\n- May 24 12:01:10 host sshd[12405]: Failed password for root from 192.168.1.150 port 54108 ssh2\n- May 24 12:01:14 host sshd[12410]: Accepted publickey for backup_admin from 192.168.1.150 port 54112 ssh2: RSA SHA256:xyz...",
            "prev_action": "Load auth.log data.",
            "thought": "I am looking for authentication anomalies. Scanning the auth log, I see failed password attempts for invalid users 'admin' and 'root' from IP `192.168.1.150` followed immediately by an accepted publickey authentication for 'backup_admin' from the same IP. This rapid transition from brute-forcing to successful login suggests a potential credential compromise or authorized audit. I should investigate the ssh keys configured for 'backup_admin'.",
            "next_action": "inspect backup_admin authorized_keys"
        },
        {
            "task": "Investigate SSH authentication anomalies in secure host logs.",
            "context": "/home/backup_admin/.ssh/authorized_keys:\n- ssh-rsa AAAAB3Nza... backup_service_key_2024 (Comment: Created by Devops team)\n- ssh-rsa AAAAB3Nza... temp_key_ext (Comment: Added by admin_temp on May 23) (Warning: unauthorized modification suspected)",
            "prev_action": "inspect backup_admin authorized_keys",
            "thought": "Looking at the `authorized_keys` for 'backup_admin', there is a standard key from 2024, but also a new key `temp_key_ext` added yesterday with the comment 'Added by admin_temp'. I need to verify if `admin_temp` had active permissions to modify the backup credentials. I will check the audit logs for user creation and modification history on May 23.",
            "next_action": "search audit logs for admin_temp creation"
        },
        {
            "task": "Investigate SSH authentication anomalies in secure host logs.",
            "context": "Audit Log Matches (May 23):\n- 14:15:22 [AUDIT] user 'admin_temp' created by local user 'developer_john'\n- 14:16:10 [AUDIT] 'admin_temp' granted sudo privileges without password requirement\n- 14:18:05 [AUDIT] 'admin_temp' added publickey to /home/backup_admin/.ssh/authorized_keys",
            "prev_action": "search audit logs for admin_temp creation",
            "thought": "The audit logs show that the user `admin_temp` was created by `developer_john`, immediately granted passwordless sudo access, and used to write a new SSH key into the `backup_admin` authorized keys. This is highly suspicious behavior for a standard developer account, implying either an insider threat or account compromise of `developer_john`. I should inspect active processes and login events associated with `developer_john`.",
            "next_action": "search processes developer_john"
        },
        {
            "task": "Investigate SSH authentication anomalies in secure host logs.",
            "context": "Process Table (developer_john):\n- bash (PID: 20110, ppid: 20108, TTY: pts/3) -> /usr/bin/nc -lvnp 4444 -e /bin/bash (Warning: Netcat reverse shell active)\n- python3 (PID: 20124) -> import socket,subprocess,os; s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)...",
            "prev_action": "search processes developer_john",
            "thought": "I have found the critical security breach. Under `developer_john`'s account, there is an active netcat listener `/usr/bin/nc -lvnp 4444 -e /bin/bash` along with a python script initiating a raw socket connection. This confirms that `developer_john`'s session has been hijacked to run a reverse shell, which was then used to create `admin_temp` and gain persistence. The immediate mitigation action is to terminate these compromised shell processes.",
            "next_action": "kill process 20110 20124"
        },
        {
            "task": "Investigate SSH authentication anomalies in secure host logs.",
            "context": "System Shell Output:\n- Process 20110 terminated successfully.\n- Process 20124 terminated successfully.\n- Alert: Reverse shell connections closed.",
            "prev_action": "kill process 20110 20124",
            "thought": "The malicious shells have been killed, closing the active reverse connections. Next, I must revoke the unauthorized persistence mechanism by removing the `admin_temp` account and its SSH key from the backup_admin credentials to lock out the attacker. I will start by deleting the user `admin_temp`.",
            "next_action": "delete user admin_temp"
        }
    ]
    
    for t in traces:
        builder.add_interaction(
            task=t["task"],
            context=t["context"],
            previous_action=t["prev_action"],
            thought=t["thought"],
            next_action=t["next_action"]
        )
    print(f"[AgentPipeline] Generated {len(traces)} baseline traces in {config.DATASET_PATH}.")

def run_agent_loop():
    """
    Main loop: runs active agent diagnostics, logs steps, unloads memory, 
    executes sleep cycle training, reloads the LoRA model, and validates it.
    """
    builder = LogBuffer()
    
    # 1. Populate database with synthetic high-quality heuristic traces if empty
    if not os.path.exists(config.DATASET_PATH) or os.path.getsize(config.DATASET_PATH) == 0:
        generate_synthetic_data(builder)
        
    print("\n=== STEP 1: Loading quantized base model for Active Inference ===")
    model, tokenizer = model_manager.load_base_model()
    
    print("\n=== STEP 2: Running Active Inference Task ===")
    print("Task: Scan the firewall logs for malicious port scanning behaviors.")
    
    # We will simulate the active agent recording its steps into the buffer.
    # To demonstrate the sleep-trigger, we will add 5 live agent steps to the LogBuffer.
    # Since Qwen-3B raw model might not generate the perfect format without training, 
    # the agent records its active steps into the builder.
    
    live_steps = [
        {
            "context": "Firewall Logs:\n- 20:01:05 ACCEPT TCP 192.168.1.15 -> 10.0.0.4:80\n- 20:01:06 DROP TCP 192.168.1.189 -> 10.0.0.4:21\n- 20:01:06 DROP TCP 192.168.1.189 -> 10.0.0.4:22\n- 20:01:07 DROP TCP 192.168.1.189 -> 10.0.0.4:23\n- 20:01:07 DROP TCP 192.168.1.189 -> 10.0.0.4:25\n- 20:01:08 DROP TCP 192.168.1.189 -> 10.0.0.4:80",
            "prev_action": "Initial firewall log query.",
            "thought": "I see multiple DROP logs in rapid succession originating from IP 192.168.1.189 targeting sequential ports (21, 22, 23, 25, 80) on host 10.0.0.4. This is a classic signature of a port scan. I should investigate if there are other DROP logs from this IP across the entire cluster subnet to see the scope of the scan.",
            "next_action": "search subnet drops 192.168.1.189"
        },
        {
            "context": "Subnet Search Logs (192.168.1.189):\n- 20:01:10 DROP TCP 192.168.1.189 -> 10.0.0.5:443\n- 20:01:12 DROP TCP 192.168.1.189 -> 10.0.0.6:443\n- 20:01:15 ACCEPT TCP 192.168.1.189 -> 10.0.0.12:8080 (Warning: Connection established to dev server)",
            "prev_action": "search subnet drops 192.168.1.189",
            "thought": "The subnet scan shows additional drops on hosts 10.0.0.5 and 10.0.0.6, but crucially, a successful connection was established to host 10.0.0.12 on port 8080. This dev server might have been breached or is exposing sensitive development endpoints. I must check the active services on 10.0.0.12 port 8080.",
            "next_action": "inspect services 10.0.0.12 8080"
        },
        {
            "context": "Services Report (10.0.0.12:8080):\n- Process: node (PID: 3410) -> index.js\n- Service details: Unauthenticated Swagger API playground exposed.\n- Request Logs: 20:01:18 POST /api/dev/exec - Status 200 (Payload: cat /etc/passwd)",
            "prev_action": "inspect services 10.0.0.12 8080",
            "thought": "The Node.js service running on port 8080 of 10.0.0.12 exposed an unauthenticated development execution endpoint `/api/dev/exec`. The attacker at 192.168.1.189 successfully executed `cat /etc/passwd` via a POST request, achieving Remote Code Execution (RCE). I need to check the process user and network connections for node process 3410.",
            "next_action": "inspect process 3410 credentials"
        },
        {
            "context": "Process Details (PID 3410):\n- User: root (Alert: Node service running as superuser!)\n- Est. Connections: 192.168.1.189:8080 -> 10.0.0.12:49811 (Active socket)",
            "prev_action": "inspect process 3410 credentials",
            "thought": "The vulnerable Node service is running as `root`! This means the attacker's remote code execution has full root authority over the developer host 10.0.0.12. Furthermore, there is an active connection from 192.168.1.189. I must kill Node process 3410 immediately to terminate their active root-level shell access.",
            "next_action": "kill process 3410"
        },
        {
            "context": "System shell output:\n- Process 3410 killed successfully.\n- Network connection to 192.168.1.189 severed.",
            "prev_action": "kill process 3410",
            "thought": "I have successfully terminated the compromised Node service and severed the attacker's active socket. The immediate threat is neutralised, but the Swagger endpoint must be secured. I will record this task as mitigated and complete the investigation.",
            "next_action": "complete task"
        }
    ]
    
    task_desc = "Scan firewall logs for malicious port scanning behaviors."
    buffer_count = 0
    for step in live_steps:
        buffer_count = builder.add_interaction(
            task=task_desc,
            context=step["context"],
            previous_action=step["prev_action"],
            thought=step["thought"],
            next_action=step["next_action"]
        )
        
    print(f"\n[AgentPipeline] Session buffer reached size {buffer_count}.")
    
    # 3. Check if we reached threshold to trigger the Sleep Cycle (Training)
    if buffer_count >= config.BUFFER_THRESHOLD:
        print("\n=== STEP 3: Triggering Sleep Cycle (Unloading Model & Training) ===")
        
        # Free GPU Memory from the inference model to make room for training
        print("Deleting active inference model graph...")
        del model
        model_manager.clear_gpu_memory()
        
        # Load the base model and prepare for LoRA fine-tuning
        print("Re-loading model and preparing LoRA adapters for training...")
        train_model, train_tokenizer = model_manager.load_base_model()
        train_model = model_manager.attach_mlp_lora(train_model)
        
        # Retrieve the dataset containing all baseline traces + our 5 new live traces
        dataset = builder.get_dataset()
        
        # Execute the sleep cycle training
        micro_trainer.run_sleep_cycle(train_model, train_tokenizer, dataset)
        
        # Clean up training resources from VRAM
        print("Unloading training graph...")
        del train_model
        del train_tokenizer
        model_manager.clear_gpu_memory()
        
        # 4. Load the final model with the newly trained LoRA adapter
        print("\n=== STEP 4: Reloading Base Model with Trained MLP LoRA Adapter ===")
        base_model, val_tokenizer = model_manager.load_base_model()
        
        print(f"Attaching newly trained LoRA weights from {config.OUTPUT_DIR}...")
        peft_model = PeftModel.from_pretrained(base_model, config.OUTPUT_DIR)
        
        # Put in evaluation mode
        peft_model.eval()
        print("Model reloaded successfully in evaluation mode.")
        
        # 5. Run a validation inference step to show the model uses its new cognitive heuristics!
        print("\n=== STEP 5: Verifying Cognitive Monologue Generation ===")
        validation_prompt = (
            "Task: Diagnose memory leak.\n\n"
            "Context:\n"
            "- service_web: RSS=450MB, cpu=2.1%\n"
            "- service_ml: RSS=8.9GB, cpu=95.4% (Alert: Out of RAM expected)\n\n"
            "Previous Action: None"
        )
        
        # System instructions configured during SFT
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an agentic AI system. You possess advanced cognitive heuristics to scan messy server contexts "
                    "and previous steps. Always explain your detailed internal thought process (Thought Process) showing "
                    "how and why you are searching specific parts of the context, then output your final Next Action."
                )
            },
            {"role": "user", "content": validation_prompt}
        ]
        
        # Compile inputs
        inputs = val_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
        
        print("Generating response with fine-tuned MLP LoRA model...")
        with torch.no_grad():
            outputs = peft_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=val_tokenizer.eos_token_id
            )
            
        generated_response = val_tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print("\n--- Model Response Output ---")
        print(generated_response)
        print("-----------------------------\n")
        
        # Clean up validation model
        del peft_model
        del base_model
        model_manager.clear_gpu_memory()
        
        print("[AgentPipeline] Execution pipeline ran completely and successfully!")
        
if __name__ == "__main__":
    run_agent_loop()
