# 💮 Araliya Tiny-byLLM Plugin

Seamlessly integrates byllm calls with TinyLlama through a local tiny-server, supporting fine-tuning and evaluation workflows.

---

## 🚀 Quick Start


### 🔧Tiny-byLLM(Plugin)

**Installation**
```bash
cd tiny-byllm/
pip install -e .
```

### 🖥️ Run Tiny Server

**Install dependencies:**

```bash
pip install fastapi "uvicorn[standard]" accelerate outlines transformers peft datasets
```

**Run the server:**

```bash
cd tiny-server/
uvicorn app:app --host 0.0.0.0 --port 7000
```
---

## ✨ Features

🔌 Seamless Jac integration

🧠 TinyLlama 1.1B backbone

🔄 Intelligent training & evaluation process

🌐 Independent OpenAI API–compatible tiny-server

🎛 Hot-swappable LoRA adapters (as for multitask usage — to be added)

   
---

## 🔧 Compatibility

✅ Updated to work with byllm

🗂 Use the mtllm-compatible branch for mtllm (deprecated)

---

## 🏗️ Architecture

<p align="center">
  <img src="imgs/pic1.jpg" alt="Architecture Overview" width="600"/>
</p>

The plugin intercepts byllm calls and, depending on the mode set by the tiny-server, routes inference either to a local TinyLLaMA server (running on port 7000) or to a cloud LLM.

The local TinyLLaMA server exchanges:

- Training Data / Inference results with the plugin.

- Control signals to manage fine-tuning, evaluation, and adapter usage (the available control endpoints are documented in the tiny-server README).

- The tiny-server itself is an OpenAI API–compatible server that supports constrained decoding through outlines, enabling structured output generation. This architecture allows seamless switching between local and cloud backends while ensuring flexible, reliable integration with Jac.

<p align="center">
  <img src="imgs/pic2.jpg" alt="Architecture Overview" width="600"/>
</p>


