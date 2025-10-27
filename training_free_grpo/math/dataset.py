import os
import json
import random
from typing import List, Dict, Any
from datasets import load_dataset


def load_data(name: str) -> List[Dict[str, Any]]:

    # Support local file paths
    if os.path.exists(name):
        if name.endswith(".json"):
            return json.load(open(name))
        if name.endswith(".jsonl"):
            return [json.loads(line) for line in open(name)]
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