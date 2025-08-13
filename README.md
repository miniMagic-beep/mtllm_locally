# ⚙️ mtllm_locally

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
4.  **🔀 Implement Dynamic Routing**: Enable the plugin to automatically route API requests externally to other LLMs when necessary.