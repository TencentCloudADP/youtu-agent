#!/bin/bash

set -e

# Preprocess the Asearcher JSON dataset for Parquet
PROJECT_DIR="$(pwd)"
echo $PROJECT_DIR
export DATA_ROOT_PATH="${PROJECT_DIR}/datasets/asearcher_data/"

INPUT_JSONL_PATH="${DATA_ROOT_PATH}/ASearcher-train-data/ASearcher-Base-35k.jsonl"
LOCAL_PATH="${DATA_ROOT_PATH}/ASearcher-train-data/base"

python3 preprocess_asearcher.py --input_jsonl_path $INPUT_JSONL_PATH --local_dir $LOCAL_PATH

INPUT_JSONL_PATH="${DATA_ROOT_PATH}/ASearcher-train-data/ASearcher-LRM-35k.jsonl"
LOCAL_PATH="${DATA_ROOT_PATH}/ASearcher-train-data/lrm"

python3 preprocess_asearcher.py --input_jsonl_path $INPUT_JSONL_PATH --local_dir $LOCAL_PATH

INPUT_JSONL_PATH="${DATA_ROOT_PATH}/ASearcher-test-data/"
LOCAL_PATH="${DATA_ROOT_PATH}/ASearcher-test-data/"

python3 preprocess_asearcher_test.py --input_jsonl_path $INPUT_JSONL_PATH --local_dir $LOCAL_PATH

