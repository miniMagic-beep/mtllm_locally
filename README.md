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
