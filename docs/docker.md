# Quick Start with Docker

This guide walks you through deploying Youtu-agent using Docker containers. It covers the essential steps to get your first agent running with an interactive frontend interface.


## Prerequisites

- Docker installed on your system
    - Visit <https://www.docker.com/> to download and install Docker if needed
    - Verify your installation by running `docker --version`

## Deployment Steps

### Step 1. Build Docker Image

Build the Youtu-agent Docker image:

```bash
docker build -t youtu-agent .
```

### Step 2. Configure Environment

1. Create a configuration file by copying the template:

```bash
cp .env.docker.example .env
```

2. Configure the following required variables in `.env`:

```plaintext
# LLM Configuration
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=<your-api-key>

# Serper API Configuration
# Get your key from https://serper.dev/playground
SERPER_API_KEY=<your-serper-key>

# Frontend Configuration
# Note: IP must be 0.0.0.0 for Docker container port forwarding
UTU_WEBUI_PORT=8848
UTU_WEBUI_IP=0.0.0.0
```

### Step 3. Launch the Service

#### Option 1: Run the Default Web Search Agent Demo

By replacing the `/path/to/your/.env`, please run:

```bash
docker run -it \
    -p 8848:8848 \
    -v "/path/to/your/.env:/youtu-agent/.env" \
    youtu-agent
```

The service will be accessible at <http://127.0.0.1:8848>

#### Option 2: Interactive Shell Access
To run other examples or custom configurations by replacing the `/path/to/your/.env`:

```bash
docker run -it \
    -p 8848:8848 \
    -v "/path/to/your/.env:/youtu-agent/.env" \
    youtu-agent \
    bash
```

---

## Next Steps

- **Explore Examples**: Check the `/examples` directory for more detailed use cases and advanced scripts.
