import os
import re
from datetime import datetime
from math_verify import parse, verify
import json
import yaml
import xml.etree.ElementTree as ET

DEBUG_MODE = True

def extract_key_value_pairs(text):
    """Extracts (KEY, VALUE) pairs from a string with <im_start>KEY: VALUE<im_end> segments."""
    pattern = r"<im_start>(.*?):\s*(.*?)<im_end>"
    return re.findall(pattern, text.strip())

def accuracy_infos_v0(completions, assistant, **kwargs):
    """Evaluate percentage of correctly predicted VALUEs across all pages."""
    contents = [completion[0]["content"] for completion in completions]
    solutions = [a["content"] for a in assistant]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    total_match_count = 0
    total_value_count = 0

    result_log_path = os.path.join(os.getenv("OUTPUT_DIR", "."), "results_infos.json")
    with open(result_log_path, "a", encoding="utf-8") as f:
        for pred_text, gt_text in zip(contents, solutions):
            pred_kv = dict(extract_key_value_pairs(pred_text))
            gt_kv = dict(extract_key_value_pairs(gt_text))

            match_count = 0
            total = len(gt_kv)
            total_value_count += total

            for key, gt_val in gt_kv.items():
                pred_val = pred_kv.get(key)
                if pred_val == gt_val:
                    match_count += 1

            total_match_count += match_count
            reward = match_count / total if total > 0 else 0.0
            rewards.append(reward)

            # Save each page's result
            result_record = {
                "timestamp": current_time,
                "ground_truth": gt_kv,
                "prediction": pred_kv,
                "matched_keys": match_count,
                "total_keys": total,
                "reward": reward
            }
            f.write(json.dumps(result_record, ensure_ascii=False) + "\n")

            # Optional debug logging
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH", "accuracy_infos_debug.log")
                with open(log_path, "a") as logf:
                    logf.write(f"\n=== {current_time} ===\n")
                    logf.write(f"[GT  ] {gt_kv}\n")
                    logf.write(f"[PRED] {pred_kv}\n")
                    logf.write(f"[MATCH] {match_count} / {total} → reward = {reward}\n")

    # Print or return global summary
    if total_value_count > 0:
        overall_accuracy = total_match_count / total_value_count
    else:
        overall_accuracy = 0.0

    print(f"\n✅ Overall accuracy across all VALUEs: {total_match_count} / {total_value_count} → {overall_accuracy:.2%}")
    
    return rewards

def extract_key_value_pairs_chatml(text):
    pattern = r"<im_start>(.*?):\s*(.*?)<im_end>"
    return re.findall(pattern, text.strip())

def extract_key_value_pairs_json(text):
    try:
        data = json.loads(text)
        return list(data.items())
    except Exception:
        return []

def extract_key_value_pairs_yaml(text):
    try:
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return list(data.items())
        else:
            return []
    except Exception:
        return []

def extract_key_value_pairs_xml(text):
    try:
        root = ET.fromstring(text)
        return [(child.tag, child.text or "") for child in root]
    except Exception:
        return []

def detect_format_and_extract(text):
    if "<im_start>" in text and "<im_end>" in text:
        print("ChatML format...")
        return extract_key_value_pairs_chatml(text)
    elif text.strip().startswith("{") and text.strip().endswith("}"):
        print("JSON format...")
        return extract_key_value_pairs_json(text)
    elif text.strip().startswith("<record>") and text.strip().endswith("</record>"):
        print("XML format...")
        return extract_key_value_pairs_xml(text)
    else:
        print("YAML format...")
        return extract_key_value_pairs_yaml(text)  # fallback

def accuracy_infos(completions, assistant, **kwargs):
    """Evaluate accuracy of VALUEs across multiple output formats (ChatML, JSON, YAML, XML)."""
    contents = [completion[0]["content"] for completion in completions]
    solutions = [a["content"] for a in assistant]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    total_match_count = 0
    total_value_count = 0

    result_log_path = os.path.join(os.getenv("OUTPUT_DIR", "."), "results_infos.json")
    with open(result_log_path, "a", encoding="utf-8") as f:
        for pred_text, gt_text in zip(contents, solutions):
            pred_kv = dict(detect_format_and_extract(pred_text))
            gt_kv = dict(detect_format_and_extract(gt_text))

            match_count = 0
            total = len(gt_kv)
            total_value_count += total

            for key, gt_val in gt_kv.items():
                pred_val = pred_kv.get(key)
                if pred_val == gt_val:
                    match_count += 1

            total_match_count += match_count
            reward = match_count / total if total > 0 else 0.0
            rewards.append(reward)

            result_record = {
                "timestamp": current_time,
                "ground_truth": gt_kv,
                "prediction": pred_kv,
                "matched_keys": match_count,
                "total_keys": total,
                "reward": reward
            }
            f.write(json.dumps(result_record, ensure_ascii=False) + "\n")

            if os.getenv("DEBUG_MODE") == "true":
                debug_log_path = os.path.join(os.getenv("OUTPUT_DIR", "."), "accuracy_infos_debug.json")
                with open(debug_log_path, "a") as logf:
                    logf.write(f"\n=== {current_time} ===\n")
                    logf.write(f"[GT  ] {gt_kv}\n")
                    logf.write(f"[PRED] {pred_kv}\n")
                    logf.write(f"[MATCH] {match_count} / {total} → reward = {reward:.4f}\n")
                print(f"[DEBUG] saved {debug_log_path}")

    overall_accuracy = total_match_count / total_value_count if total_value_count > 0 else 0.0
    print(f"\n✅ Overall accuracy across all VALUEs: {total_match_count} / {total_value_count} → {overall_accuracy:.2%}")
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
