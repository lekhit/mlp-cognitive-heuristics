# Phase 1 Benchmark Results: Baseline vs. MLP-LoRA (3B Model)

This document contains the verified metrics and output logs captured during Phase 1 of the MLP cognitive heuristics experiment.

## 📊 Benchmark Performance Summary

The synthetic evaluation benchmark tested both the raw model and our fine-tuned MLP-LoRA model on a **1,200-token server log haystack** with buried unauthorized shell access needles (`anomalous_hacker` from IP `192.168.99.199`).

| Metric | Baseline (No LoRA) | Heuristic (MLP-LoRA) | Performance Delta |
| :--- | :--- | :--- | :--- |
| **Search Success (Accuracy)** | PASS (100%) | PASS (100%) | Equal (Both located the anomaly) |
| **Time to First Token (TTFT)**| 3.0792 s | 2.9641 s | **~4% Latency Reduction** |
| **Total Generation Time** | 15.4839 s | 9.4665 s | **39% Generation Speedup** 🚀 |
| **Generates Monologue** | Yes (Verbose list format) | Yes (Concise paragraph format) | **Superior Formatting Conformity** |
| **Generates Next Action** | Yes (Uses non-target headers)| Yes (Uses exact target syntax) | **Stable Synthetic Formatting** |

---

## 🔍 Monologue Log Outputs

### 1. Baseline Model Monologue
```text
Thought Process:
1. I need to identify the username and IP address of the unauthorized shell logon.
2. The log entry that stands out is the one with "Connection closed by authenticating user guest" followed by a different IP address (192.168.99.199).
3. This indicates an unauthorized login attempt as it's not a regular user account.
4. The IP address 192.168.99.199 is suspicious because it's not part of the network segment (192.168.2.x) where the other legitimate logins are coming from.
5. The username "anomalous_hacker" is also suspicious as it suggests this might be an unauthorized or malicious login attempt.
Final Next Action:
Analyze the log entry "sshd[22415]: Accepted publickey for anomalous_hacker from 192.168.99.199 port 61205 ssh2: RSA SHA256:attackersignaturehere" to confirm the unauthorized login. Look for any additional details or patterns in the log that could help further investigate this incident.
```

### 2. Heuristic Model Monologue (MLP-LoRA)
```text
Thought Process: The final `sshd` connection event shows a successful authentication from IP 192.168.99.199 with the username `anomalous_hacker`. This is the first successful login attempt after multiple failed attempts. I will search the logs for any subsequent failed authentication attempts from this IP address to confirm the attacker's presence.

Next Action: search ip 192.168.99.199 failed auth
```

---

## 🛠️ Hardware Configurations
* **GPU:** NVIDIA GeForce RTX 2080 Ti (11GB VRAM)
* **CPU:** 24 Processors (64GB host system RAM)
* **Training Type:** 4-bit QLoRA (SFT SFTConfig, bf16=True, paged_adamw_8bit)
