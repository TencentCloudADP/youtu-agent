# Environment Variables

This document outlines the configuration of key environment variables in Youtu-Agent.

## Overview

Before running Youtu-Agent, you need to configure the necessary environment variables. A recommended practice is to start by copying the `.env.example` file from the project root:

```sh
cp .env.example .env
```

Then, edit the `.env` file to fill in the required API Keys and other configurations. For a more comprehensive list of options, please refer to the [`.env.full`](https://github.com/TencentCloudADP/youtu-agent/blob/main/.env.full) file, which contains all available environment variables and their detailed descriptions.

## LLM API Keys

The core capabilities of Youtu-Agent rely on Large Language Models (LLMs). You need to configure the appropriate models for different functional modules.

### Core LLM

This is the primary text generation LLM that the Agent relies on for its operations.

```sh
# LLM type (e.g., chat.completions, responses)
UTU_LLM_TYPE=chat.completions
# LLM model name (e.g., deepseek-chat, gpt-4-turbo)
UTU_LLM_MODEL=deepseek-chat
# LLM provider's API base URL (e.g., https://api.deepseek.com/v1)
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
# LLM provider's API Key
UTU_LLM_API_KEY=YOUR_API_KEY
```

### Vision and Audio LLMs

When using the `ImageToolkit` or `AudioToolkit`, you need to configure LLMs that support multimodal capabilities.

**Image (Vision) LLM:**

```sh
# Image LLM type
UTU_IMAGE_LLM_TYPE=
# Image LLM model name (e.g., qwen-vl-plus)
UTU_IMAGE_LLM_MODEL=
# Image LLM API base URL
UTU_IMAGE_LLM_BASE_URL=
# Image LLM API Key
UTU_IMAGE_LLM_API_KEY=
```

**Audio LLM:**

```sh
# Audio LLM model name (e.g., whisper-1)
UTU_AUDIO_LLM_MODEL=
# Audio LLM API base URL
UTU_AUDIO_LLM_BASE_URL=
# Audio LLM API Key
UTU_AUDIO_LLM_API_KEY=
```

### Evaluation LLM

When running the evaluation framework (`/utu/eval`), a separate "Judge" LLM is required to score and assess the Agent's output.

```sh
# Judge LLM type
JUDGE_LLM_TYPE=
# Judge LLM model name
JUDGE_LLM_MODEL=
# Judge LLM API base URL
JUDGE_LLM_BASE_URL=
# Judge LLM API Key
JUDGE_LLM_API_KEY=
```

## Tools

Some Toolkits require their own API Keys or specific configurations.

### Search Toolkit (`SearchToolkit`)

The search toolkit integrates the following two services by default:

1. **Web Search**: Uses the efficient Google Search API provided by [Serper](https://serper.dev/). You will need to register and obtain an API Key.
2. **Web Content Extraction**: Uses the [Jina AI Reader](https://jina.ai/reader/) to convert web page content into an LLM-friendly Markdown format. This also requires registration and an API Key.

Configure them in your `.env` file as follows:

```sh
# Get from https://serper.dev/
SERPER_API_KEY=YOUR_SERPER_API_KEY
# Get from https://jina.ai/reader/
JINA_API_KEY=YOUR_JINA_API_KEY
```

## Tracing & Monitoring

The framework integrates [OpenTelemetry](https://opentelemetry.io/) and [Phoenix](https://arize.com/docs/phoenix) for tracing and monitoring the Agent's execution flow.

```sh
# Phoenix/OpenTelemetry trace receiver endpoint
PHOENIX_ENDPOINT=http://127.0.0.1:6006/v1/traces
# Project name displayed in the Phoenix UI
PHOENIX_PROJECT_NAME=Youtu-Agent
```

## Miscellaneous

### Hugging Face

Used to download datasets (e.g., GAIA) from the Hugging Face Hub.

```sh
# Get from https://huggingface.co/settings/tokens
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

### Web UI

Used to configure the address and port for the front-end interface.

```sh
# Port for the frontend service to listen on
UTU_WEBUI_PORT=8848
# IP address for the frontend service to listen on
UTU_WEBUI_IP=127.0.0.1
# Whether to enable auto reload for tornado server
UTU_WEBUI_AUTOLOAD=false
```
