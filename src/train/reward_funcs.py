import os
import re
from datetime import datetime
from math_verify import parse, verify
import json

def extract_key_value_pairs(text):
    """Extracts (KEY, VALUE) pairs from a string with <im_start>KEY: VALUE<im_end> segments."""
    pattern = r"<im_start>(.*?):\s*(.*?)<im_end>"
    return re.findall(pattern, text.strip())

def accuracy_infos(completions, assistant, **kwargs):
    """Reward function that compares only the VALUEs of KEY: VALUE pairs between predicted and ground-truth completions."""
    contents = [completion[0]["content"] for completion in completions]
    solutions = [a["content"] for a in assistant]
    rewards = []
    detailed_logs = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for pred_text, gt_text in zip(contents, solutions):
        pred_kv = dict(extract_key_value_pairs(pred_text))
        gt_kv = dict(extract_key_value_pairs(gt_text))

        match_count = 0
        total = len(gt_kv)

        for key, gt_val in gt_kv.items():
            pred_val = pred_kv.get(key)
            if pred_val == gt_val:
                match_count += 1

        # reward: percentage of correct values
        reward = match_count / total if total > 0 else 0.0
        rewards.append(reward)

        # ✅ Save result
        result_log_path = os.path.join(os.getenv("OUTPUT_DIR", "."), "results_infos.json")
        result_record = {
            "timestamp": current_time,
            "ground_truth": gt_kv,
            "prediction": pred_kv,
            "matched_keys": match_count,
            "total_keys": total,
            "reward": reward
        }
        with open(result_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_record, ensure_ascii=False) + "\n")

        # Optional debug log
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "accuracy_infos_debug.log")
            with open(log_path, "a") as f:
                f.write(f"\n=== {current_time} ===\n")
                f.write(f"[GT  ] {gt_kv}\n")
                f.write(f"[PRED] {pred_kv}\n")
                f.write(f"[MATCH] {match_count} / {total} → reward = {reward}\n")

    return rewards


def accuracy_reward(completions, assistant, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    solution = [a['content'] for a in assistant]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>(.*?)</answer>", sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)

         # ✅ Save result
        result_log_path = os.path.join(os.getenv("OUTPUT_DIR", "."), "results_all.json")
        result_record = {
            "timestamp": current_time,
            "ground_truth": sol,
            "prediction": content,
            "reward": reward
        }
        with open(result_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
