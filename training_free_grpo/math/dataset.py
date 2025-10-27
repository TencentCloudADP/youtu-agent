import os
import json
import random
from typing import List, Dict, Any
import pandas as pd
from datasets import load_dataset


def load_data(name: str) -> List[Dict[str, Any]]:

    # Support local file paths
    if os.path.exists(name):
        if name.endswith(".json"):
            data = json.load(open(name))
            return _normalize_local_math_records(data)
        if name.endswith(".jsonl"):
            data = [json.loads(line) for line in open(name)]
            return _normalize_local_math_records(data)
        if name.endswith(".parquet"):
            df = pd.read_parquet(name)
            data = df.to_dict(orient="records")
            return _normalize_local_math_records(data)
        raise ValueError(f"Unsupported local dataset format: {name}")

    if name == "AIME24":    
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        data = [{"problem": each["problem"], "groundtruth": each["answer"]} for each in dataset.to_list()]
        return data

    elif name == "AIME25":
        dataset = load_dataset("yentinglin/aime_2025", split="train")
        data = [{"problem": each["problem"], "groundtruth": each["answer"]} for each in dataset.to_list()]
        return data
    
    elif name == "DAPO-Math-17k":
        if os.path.exists("data/math/dataset/DAPO-Math-17k.json"):
            return json.load(open("data/math/dataset/DAPO-Math-17k.json"))
        else:
            dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
            data = dataset.to_list()
            transformed = {}
            for record in data:
                problem = record["prompt"][0]["content"].replace(
                    "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n",
                    "",
                ).replace('\n\nRemember to put your answer on its own line after "Answer:".', "")
                groundtruth = record["reward_model"]["ground_truth"]
                transformed[problem] = groundtruth
            random.seed(42)
            transformed = [{"problem": k, "groundtruth": v} for k, v in transformed.items()]
            random.shuffle(transformed)
            os.makedirs("data/math/dataset", exist_ok=True)
            json.dump(transformed, open("data/math/dataset/DAPO-Math-17k.json", "w"), indent=2)
            return transformed

    raise ValueError(f"Unsupported dataset: {name}")


def _normalize_local_math_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize various math dataset schemas to {problem, groundtruth, golden_answer?}.

    Supports GSM8K-style columns: question, answer (answer may contain "#### final").
    Passes through records already containing problem/groundtruth.
    """
    normalized: List[Dict[str, Any]] = []
    for rec in records:
        if "problem" in rec and "groundtruth" in rec:
            normalized.append(rec)
            continue
        problem = rec.get("question") or rec.get("prompt") or rec.get("problem")
        answer = rec.get("groundtruth") or rec.get("golden_answer") or rec.get("answer") or rec.get("final_answer")
        if isinstance(answer, str) and "####" in answer:
            # GSM8K answers often like: "...\n#### 42"
            try:
                answer = answer.split("####", 1)[-1].strip()
            except Exception:
                pass
        if problem is None or answer is None:
            # Skip rows that cannot be normalized
            continue
        out = {"problem": problem, "groundtruth": answer}
        # If provided, keep explicit golden answer
        if "golden_answer" in rec:
            out["golden_answer"] = rec["golden_answer"]
        normalized.append(out)
    return normalized