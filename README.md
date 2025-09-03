# 💮 Araliya Tiny-byLLM Plugin

Redirects `byllm` calls to TinyLLaMA.

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