import os
import json
import pandas as pd
from datasets import load_dataset


def load_data(dataset_name):
    """ dataset_name: {dataset}_{sample_number} """
    # Support local file paths (json/jsonl/parquet)
    if os.path.exists(dataset_name):
        if dataset_name.endswith(".json"):
            return json.load(open(dataset_name))
        if dataset_name.endswith(".jsonl"):
            return [json.loads(line) for line in open(dataset_name)]
        if dataset_name.endswith(".parquet"):
            import pandas as pd
            return pd.read_parquet(dataset_name).to_dict(orient="records")
        raise ValueError(f"Unsupported local dataset format: {dataset_name}")
    if dataset_name.startswith("AFM_web_RL"):
        data = load_AFM_web_RL()
    elif dataset_name.startswith("WebWalkerQA"):
        data = load_WebWalkerQA()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # check if sampled dataset
    if dataset_name.rsplit("_", 1)[-1].isdigit():
        n = int(dataset_name.rsplit("_", 1)[-1])
        print(f"- Sampling {n} examples from {dataset_name.rsplit('_', 1)[0]}")
        if dataset_name.startswith("AFM_web_RL"):
            data = sampling_AFM_web_RL(data, n=n)
        elif dataset_name.startswith("WebWalkerQA"):
            data = sampling_WebWalkerQA(data, n=n)
        print(f"- Sampled {len(data)} examples")
    
    return data


def load_AFM_web_RL(split="train"):
    dataset = load_dataset("PersonalAILab/AFM-WebAgent-RL-Dataset", split=split)
    data = []
    for i, row in enumerate(dataset.to_list()):
        if len(row["extra_info"]["answer"]) > 1:
            continue
        data.append(
            {
                "id": i + 1,
                "problem": row["extra_info"]["question"],
                "groundtruth": row["extra_info"]["answer"][0],
            }
        )
    print(f"Total {len(data)} samples")
    print(f"Sample: {data[0]}")
    return data


def sampling_AFM_web_RL(data, n, random_seed=42):
    """ Sample n examples from AFM_web_RL dataset. """

    df = pd.DataFrame(data)
    df_sampled = df.sample(n=n, random_state=random_seed).reset_index(drop=True)
    df_sampled["index"] = range(1, len(df_sampled) + 1)
    # gen new id and save the original id to source_id
    df_sampled["source_id"] = df_sampled["id"]
    df_sampled["id"] = range(1, len(df_sampled) + 1)
    
    return df_sampled.to_dict(orient="records")


def load_WebWalkerQA(split="main"):
    dataset = load_dataset("callanwu/WebWalkerQA", split=split)
    data = []
    for i, row in enumerate(dataset.to_list()):
        level_map = {
            "easy": 1,
            "medium": 2,
            "hard": 3,
        }
        data.append(
            {
                "id": i + 1,
                "problem": row["question"],
                "groundtruth": row["answer"],
                "level": level_map[row["info"]["difficulty_level"]],
                "root_url": row["root_url"],
                "info": json.dumps(row["info"], ensure_ascii=False),
            }
        )
    print(f"Total {len(data)} samples")
    print(f"Sample: {data[0]}")
    return data


def sampling_WebWalkerQA(data, n, random_seed=42):
    """ Sample n examples from WebWalkerQA with the rate of difficulty keep the same. """
    difficluty_ratio = {1: 4, 2: 7, 3: 6}  # easy:medium:hard = 4:7:6
    total_ratio = sum(difficluty_ratio.values())
    n_easy = n * difficluty_ratio[1] // total_ratio
    n_medium = n * difficluty_ratio[2] // total_ratio
    n_hard = n * difficluty_ratio[3] // total_ratio
    assert n_easy + n_medium + n_hard == n, "The sample number of WebWalkerQA must be multiple of 17."

    df = pd.DataFrame(data)
    df_sampled = pd.concat(
        [
            df[df["level"] == 1].sample(n=n_easy, random_state=random_seed),
            df[df["level"] == 2].sample(n=n_medium, random_state=random_seed),
            df[df["level"] == 3].sample(n=n_hard, random_state=random_seed),
        ]
    ).reset_index(drop=True)
    df_sampled["index"] = range(1, len(df_sampled) + 1)
    # gen new id and save the original id to source_id
    df_sampled["source_id"] = df_sampled["id"]
    df_sampled["id"] = range(1, len(df_sampled) + 1)
    
    return df_sampled.to_dict(orient="records")


def save_dataset(dataset_name, data):
    dataset_dir = os.path.join("data", "dataset2")
    os.makedirs(dataset_dir, exist_ok=True)

    df = pd.DataFrame(data)
    dataset_filename = os.path.join(dataset_dir, f"{dataset_name}.parquet")
    df.to_parquet(dataset_filename, index=False)
    print(f"Saved {len(data)} samples to {dataset_filename}")


if __name__ == "__main__":
    # directly load data from huggingface with sampling
    load_data("AFM_web_RL_100")

    # you can also run the following code to build and sample the dataset
    # train_dataset_name = "AFM_web_RL"
    # eval_dataset_name = "WebWalkerQA"
    # # 1. load data from huggingface
    # train_data = load_WebWalkerQA()
    # eval_data = load_WebWalkerQA()
    # save_dataset(train_dataset_name, train_data)
    # save_dataset(eval_dataset_name, eval_data)
    # # 2. sample n=51 from WebWalkerQA data
    # data_sampled = sampling_WebWalkerQA(eval_data, n=51)
    # save_dataset(f"{eval_dataset_name}_51", data_sampled)
    # # 3. sample n=100 from AFM_web_RL data
    # data_sampled = sampling_AFM_web_RL(train_data, n=100)
    # save_dataset(f"{train_dataset_name}_100", data_sampled)