# mtllm_locally

# Local LLM Plugin for Jaclang

A custom MTLLM plugin and local server to integrate a specialized TinyLlama model into Jaclang's `by llm()` workflow.

***

## Current Status -

* **Local LLM Server**: A proxy server using `llama.cpp` and `LiteLLM` is running TinyLlama locally, making the model available via an API endpoint.
* **Custom MTLLM Plugin**: A Jaclang plugin successfully intercepts `by llm()` calls and routes them to the local server for processing.

***

## Next Steps -

* **Fine-Tune Model**: Create a domain-specific dataset to fine-tune TinyLlama, specializing it for Jaclang's unique structured data tasks.
* **Integrate Final Model**: Replace the base TinyLlama in the local proxy with the newly fine-tuned version to complete the integration.
