import json
import os
from datasets import Dataset
import config

class LogBuffer:
    """Manages recording active interactions and compiling them into SFT-compatible datasets."""
    def __init__(self):
        self.buffer = []

    def add_interaction(self, task, context, previous_action, thought, next_action):
        """Formats and logs a single cognitive step into the buffer and appends it to the local dataset."""
        interaction = {
            "messages": [
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
                },
                {
                    "role": "assistant",
                    "content": f"Thought Process: {thought}\n\nNext Action: {next_action}"
                }
            ]
        }
        
        self.buffer.append(interaction)
        
        # Append to persistent JSONL file
        with open(config.DATASET_PATH, "a") as f:
            f.write(json.dumps(interaction) + "\n")
            
        print(f"[HeuristicBuilder] Recorded interaction trace. Session buffer size: {len(self.buffer)}")
        return len(self.buffer)

    def clear_session_buffer(self):
        """Clears current session buffer (historical files remain saved)."""
        self.buffer = []

    def get_dataset(self):
        """Reads all recorded traces from JSONL and loads them into a Hugging Face Dataset."""
        traces = []
        if os.path.exists(config.DATASET_PATH):
            with open(config.DATASET_PATH, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            traces.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        
        # Fallback to session buffer if file not created
        if not traces and self.buffer:
            traces = list(self.buffer)
            
        # Fallback to standard dummy entry if absolutely no data exists (to prevent SFT trainer from crashing)
        if not traces:
            traces = [
                {
                    "messages": [
                        {"role": "system", "content": "You are an agentic AI system."},
                        {"role": "user", "content": "Task: Verify environment.\n\nContext:\nNo context.\n\nPrevious Action: None"},
                        {"role": "assistant", "content": "Thought Process: The system is running correctly and ready to receive instructions.\n\nNext Action: idle"}
                    ]
                }
            ]
            
        return Dataset.from_list(traces)
