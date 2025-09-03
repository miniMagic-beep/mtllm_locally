# Araliya MTLLM Plugin for tinyllama

file structe
my-mtllm-plugin has the jac main jac and the custom plugin
tiny-server - has the tinyillama server by OpenAI compateble server and controlls.

Refer Readme inside them for deplyemnt details and depenecides
├─ tiny-server/             # OpenAI-compatible API server powered by TinyLlama
│  ├─ app.py                # exposes `app` for ASGI
│  └─ README.md             # (add deps & install steps here)
│
├─ my-mtllm-plugin/         # JAC entrypoint + custom MTLLM plugin logic
│  ├─ main.jac
│  └─ README.md             # (add deps & install steps here)
│
└─ research/                # Research notes, experiments, and personal materials
                           # (not required for deployment)

### Local LLM Plugin for Jaclang

A custom MTLLM plugin and local server to integrate a specialized and self-improving **TinyLlama** model into Jaclang's `by llm()` workflow.

***

## 🟢 Current Status

* **🖥️ Local LLM Server**: A local server is running using **`llama.cpp`**, which serves the base **TinyLlama** model and makes it available via an API endpoint.
* **🔌 Custom MTLLM Plugin**: A Jaclang plugin uses **`LiteLLM`** to successfully intercept `by llm()` calls and route them to the local server for processing.

***

## 🚀 Next Steps

1.  **🧠 Dynamic Fine-Tuning**: Enhance the plugin to dynamically trigger fine-tuning jobs on the **TinyLlama** model using collected data, creating specialized versions automatically.
2.  **🔄 Automated Model Integration**: Implement logic for the plugin to seamlessly switch to the newly fine-tuned model versions without manual intervention.
3.  **🧪 End-to-End Testing**: Conduct thorough testing of the complete workflow, including the dynamic training and model-swapping loop.
