# Changelog

## v0.1.4 - 2026-11-06
- security(webui): fix unauthenticated remote code execution in `WebUIAgents` (CNVD report). `switch_agent` only loads configs listed by `list_agents`; `/ws` and `/upload` reject cross-origin requests (`UTU_WEBUI_ALLOWED_ORIGINS` to allow more); upload filenames are confined to the workspace; access token `UTU_WEBUI_TOKEN`, required when listening on a non-loopback address (**breaking for Docker**: set `UTU_WEBUI_TOKEN` in `.env`, or `UTU_WEBUI_ALLOW_INSECURE=true` behind an authenticating proxy). Docker docs now publish the port on `127.0.0.1` only. Upgrade if your WebUI is reachable from other hosts.

## v0.1.3 - 2025-10-31
- analysis ui: add trajectories tab by @qiuchaofan
- feat: add hooks (examples) by @Lightblues
- examples: add ppt-gen by @fpg2012 RAG example by @Lightblues
- train: add training-free GRPO branch by @pakchoi-i @YZ-Cai
- train: support rl-train @Lightblues

## v0.1.2 - 2025-10-13
- example: add ppt-gen by @fpg2012
- analysis ui: unify env variable; fix filter by tool @chaofanqiu
- ui: add file upload by @fpg2012
- example: add examples for rl @Lightblues
- tracing: add doc for phenix @Lightblues
- feat: add OrchestratorAgent @Lightblues
- enhance: refactor AynscBaseToolkit @Lightblues

## v0.1.1 - 2025-09-22
- update toolkit util scripts by @Lightblues
- LiteLLM support by @Lightblues
- Streaming support for OrchestraAgent by @fpg2012
- feat: tool maker @Lightblues
- ui: i18n, demo_universal by @fpg2012

## v0.1.0
- add `exp_analysis` frontend by @qiuchaofan
- add `CONTRIBUTING.md` by @Lightblues
- add detailed quickstart doc by @pakchoi-i
- add Docker support by @YZ-Cai

## frontend/v0.1.5 - 2025-08-29
- add the `utu_agent_ui` package by @fpg2012
