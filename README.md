# 💮 Araliya MTLLM Plugin for tinyllama

A custom [Jaseci](https://jaseci.org/) plugin to extend `tinyllama` with specialized multi-task language learning (MTLLM) capabilities. This project provides both the Jaseci (`jac`) implementation and a compatible, self-contained `tinyllama` server.

---

## Project Structure

araliya-mtllm/
├── my-mtllm-plugin/      # The core Jaseci plugin and main jac application
├── tiny-server/          # OpenAI-compatible tinyllama server and controls
└── other/                # (Research materials - not required for deployment)