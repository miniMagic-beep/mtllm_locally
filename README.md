# ⚙️ mtllm_locally

Seamlessly integrates `byllm` calls with **TinyLlama** through a local tiny-server, supporting fine-tuning and evaluation workflows.

### Local LLM Plugin for Jaclang  
A custom MTLLM plugin and local server to integrate a specialized and self-improving **TinyLlama** model into Jaclang's `by llm()` workflow.

---

## ✨ Features

- 🔌 **Seamless Jac integration** – Plug directly into Jaclang’s `by llm()` calls without extra setup.  
- 🧠 **TinyLlama 1.1B backbone** – Lightweight, efficient model designed for local execution.  
- 🔄 **Intelligent training & evaluation process** – Collects prompts, fine-tunes locally, and switches to the updated model after evaluation passes.  
- 🌐 **Independent tiny-server** – Runs a local `llama.cpp`-powered server with OpenAI API–compatible endpoints for easy integration.
- 🎛 **Hot-swappable LoRA adapters** *(planned)* – Enables multitask usage with flexible adapter switching.  

---

## 🔧 Compatibility

- ✅ **Updated for `byllm`** – Fully supports the latest Jaclang LLM plugin system.  
- 🗂 **Legacy support** – Use the `mtllm-compatible` branch for older MTLLM workflows (deprecated).  

---

## 🟢 Current Status

- **🖥️ Local LLM Server**: A local server is running using **`llama.cpp`**, serving the base **TinyLlama** model via API endpoint.  
- **🔌 Custom MTLLM Plugin**: A Jaclang plugin uses **`LiteLLM`** to intercept `by llm()` calls and route them to the local server.  

---

## ⚡ How It Works

1. Prompts are first routed to a **global model** (Gemini) to generate responses.  
2. Collected data (≈100 samples) is used to fine-tune the **TinyLlama** model locally.  
3. After fine-tuning, the local model’s outputs are **evaluated** against Gemini’s responses.  
4. ✅ If evaluation passes → the system switches to the local TinyLlama for serving requests.  
5. 🔄 This cycle repeats, enabling **continuous self-improvement**.  

---

## ▶️ Running the Project

1. **Start the local server**  
   ```bash
   ./tinyllama-server --model tinyllama-v1.0.gguf --port 8080
   ```
   
2. **Run Jaclang with the plugin**
   ```bash
   jac run main.jac
   ```

3. **Configure API endpoints** in ```config.json``` to ensure both global (Gemini) and local (TinyLlama server) models are accessible.
