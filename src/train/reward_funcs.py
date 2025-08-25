import os
import re
from datetime import datetime
from math_verify import parse, verify
import json
import yaml
import xml.etree.ElementTree as ET

def extract_key_value_pairs(text):
    """
    Extract (key, value) pairs from ChatML text <im_start>KEY: VALUE<im_end>.

    Args:
        text (str): Input text.

    Returns:
        list[tuple[str,str]]: Extracted pairs.
    """

    pattern = r"<im_start>(.*?):\s*(.*?)<im_end>"
    return re.findall(pattern, text.strip())

def accuracy_infos(completions, assistant, **kwargs):
    """
    Compute exact match accuracy between predictions and ground truth VALUEs.

    Args:
        completions (list): Model outputs, chat-style.
        assistant (list): Ground-truth references.
    
    Returns:
        list[float]: Reward per sample (0–1).
    """
    
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
    """
    Extract (key, value) pairs from ChatML text <im_start>KEY: VALUE<im_end>.
    
        Args:
            text (str): Input text.
    
        Returns:
            list[tuple[str,str]]: Extracted pairs.
    """
    
    pattern = r"<im_start>(.*?):\s*(.*?)<im_end>"
    return re.findall(pattern, text.strip())

def extract_key_value_pairs_json(text):
    """
    Extract (key, value) pairs from JSON format text.
    
        Args:
            text (str): Input text.
    
        Returns:
            list[tuple[str,str]]: Extracted pairs.
    """
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            print("⚠️ JSON parsed but is not a dict:", repr(data))
            return {}
        return data
    except Exception as e:
        print(f"❌ JSON parsing failed: {e}")
        return {}

def extract_key_value_pairs_yaml(text):
    """
    Extract (key, value) pairs from YAML format text.
    
        Args:
            text (str): Input text.
    
        Returns:
            list[tuple[str,str]]: Extracted pairs.
    """
    
    try:
        # Try to find the assistant message (in ChatML-like content)
        if "assistant" in text:
            parts = text.split("assistant", 1)
            text = parts[1].strip()

        # Remove any leading garbage lines before YAML
        lines = text.splitlines()
        yaml_lines = [line for line in lines if ':' in line and not line.strip().startswith('<')]
        yaml_text = "\n".join(yaml_lines)

        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            print("⚠️ YAML parsed but is not a dict:", repr(data))
            return {}
        return data

    except Exception as e:
        print(f"❌ YAML parsing failed: {e}")
        return {}




def extract_key_value_pairs_xml(text):
    """
    Extract (key, value) pairs from XML format text.
    
        Args:
            text (str): Input text.
    
        Returns:
            list[tuple[str,str]]: Extracted pairs.
    """
    
    try:
        root = ET.fromstring(text)
        data = {}
        for child in root.iter():
            if child is not root:
                data[child.tag] = child.text
        return data
    except ET.ParseError as e:
        print(f"❌ XML parsing failed: {e}")
        return {}
    except Exception as e:
        print(f"❌ Unknown XML error: {e}")
        return {}
        
def detect_format_and_extract(text):
    if "<im_start>" in text and "<im_end>" in text:
        print("ChatML format...")
        return "ChatML", extract_key_value_pairs_chatml(text)
    elif text.strip().startswith("{") and text.strip().endswith("}"):
        print("JSON format...")
        return "JSON", extract_key_value_pairs_json(text)
    elif text.strip().startswith("<record>") and text.strip().endswith("</record>"):
        print("XML format...")
        return "XML", extract_key_value_pairs_xml(text)
    else:
        print("YAML format...")
        return "YAML", extract_key_value_pairs_yaml(text)

def accuracy_infos_v1(completions, assistant, **kwargs):
    """Same as accuracy_infos but supports ChatML, JSON, YAML, XML formats."""

    contents = [completion[0]["content"] for completion in completions]
    solutions = [a["content"] for a in assistant]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    total_match_count = 0
    total_value_count = 0

    result_log_path = os.path.join(os.getenv("OUTPUT_DIR", "."), "results_infos.json")
    with open(result_log_path, "a", encoding="utf-8") as f:
        for pred_text, gt_text in zip(contents, solutions):
            format_type_pred, pred_kv_pairs = detect_format_and_extract(pred_text)
            format_type_gt, gt_kv_pairs = detect_format_and_extract(gt_text)
            pred_kv = dict(pred_kv_pairs)
            gt_kv = dict(gt_kv_pairs)

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

            print(f"\n=== {current_time} ===")
            print(f"[PRED FORMAT] {format_type_pred}")
            print(f"[GT FORMAT]   {format_type_gt}")
            print(f"[GT  ] {gt_kv}")
            print(f"[PRED] {pred_kv}")
            print(f"[MATCH] {match_count} / {total} → reward = {reward:.4f}")

    overall_accuracy = total_match_count / total_value_count if total_value_count > 0 else 0.0
    print(f"\n✅ Overall accuracy across all VALUEs: {total_match_count} / {total_value_count} → {overall_accuracy:.2%}")
    return rewards

def accuracy_reward(completions, assistant, **kwargs):
    """
    Reward: verify numeric/math answers (via symbolic parse/verify) else string match.
    Reward function that checks if the completion is correct using either symbolic verification or exact string matching.
    """
    
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
