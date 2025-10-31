"""
Get phoenix span url by trace_id
"""

import argparse

import tqdm

from utu.config import ConfigLoader
from utu.eval import DBDataManager
from utu.tracing.phoenix_utils import PhoenixUtils

PROJECT_NAME = "uTu-WebWalker"
phoenix_utils = PhoenixUtils(project_name=PROJECT_NAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="ww", help="Configuration name for evaluation.")
    parser.add_argument("--exp_id", type=str, default=None, help="Experiment ID.")
    args = parser.parse_args()

    config = ConfigLoader.load_eval_config(args.config_name)
    if args.exp_id:
        config.exp_id = args.exp_id

    db_manager = DBDataManager(config)
    samples = db_manager.get_samples()
    samples_wo_url = [sample for sample in samples if not sample.trace_url]
    print(f"Total samples: {len(samples)}")
    print(f"Samples without url: {len(samples_wo_url)}")
    num_updated = 0
    trace_ids_not_found = []
    for sample in tqdm.tqdm(samples_wo_url):
        if not sample.trace_url:
            phoenix_url = phoenix_utils.get_trace_url_by_id(sample.trace_id)
            if not phoenix_url:
                trace_ids_not_found.append(sample.trace_id)
                continue
            sample.trace_url = phoenix_url
            db_manager.save(sample)
            num_updated += 1
    print(f"Updated {num_updated} samples.")
    print(f"Trace IDs not found: {trace_ids_not_found}")


if __name__ == "__main__":
    main()
