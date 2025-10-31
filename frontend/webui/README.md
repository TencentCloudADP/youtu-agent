# utu_agent_ui

Static frontend pages for utu-agent `WebUIChatbot`

## Installation

1. Install npm packages

```bash
npm install
```

2. Build Python wheel

```bash
uv sync
source .venv/bin/activate
bash ./build.sh
```

You will get a wheel file in the `build` directory.

3. Install the wheel file

```bash
# install `utu_agent_ui` in the parent directory
cd ../../..
source .venv/bin/activate
uv pip install ./utu/ui/frontend/build/utu_agent_ui-0.1.5-py3-none-any.whl
```

Now the `WebUIChatbot` should work.

## Development

After `npm install`, start the development server.

```bash
npm run dev
```
