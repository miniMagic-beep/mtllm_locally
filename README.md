# mtllm_locally

**My Custom Plugin for Jaclang's MTLLM**

This plugin is designed to extend and modify the behavior of Jaclang's `by llm()` feature. It provides a simple, yet powerful, example of how to implement a custom hook to intercept and customize LLM calls. This is particularly useful for projects that require fine-tuning a model for specific, structured data tasks, such as those related to the Jac programming language.

***

## What It Does

This plugin implements the core `call_llm` hook, which allows it to intercept any call to an LLM made from a Jaclang program. It demonstrates the ability to:
- **Intercept Calls:** The plugin will print a message to the console every time a Jaclang program makes a call to an LLM.
- **Access Arguments:** It can access and print the arguments (`args`) and the `caller` function's name that were passed to the LLM.
- **Delegate to the Original Model:** It shows how to perform custom logic and then "delegate" the request to the original model for the actual response, preserving the default functionality.

This provides a foundational example for more advanced plugins, such as those for caching, logging, or even implementing a custom model provider with a fine-tuned LLM like TinyLLaMA.

***

## Installation

To use this plugin, you need to set up the package and install it in your environment.

### Prerequisites
- Python 3.11 or higher
- `mtllm` library
- `jaclang` library

### Steps

1.  **Set up the project structure:** Create the following file and folder structure.

    ```
    my-mtllm-plugin/
    ├── pyproject.toml
    └── my_mtllm_plugin/
        ├── __init__.py
        └── plugin.py
    ```

2.  **Add the plugin code:** Place the provided Python code into `my_mtllm_plugin/plugin.py`.

3.  **Configure `pyproject.toml`:** Add the following to your `pyproject.toml` file to register the plugin using a Poetry entry point.

    ```toml
    [tool.poetry]
    name = "my-mtllm-plugin"
    version = "0.1.0"
    description = "My custom MTLLM plugin"
    authors = ["Your Name <your.email@example.com>"]

    [tool.poetry.dependencies]
    python = "^3.11"
    mtllm = "*"
    jaclang = "*"

    [tool.poetry.plugins."jac"]
    my-mtllm-plugin = "my_mtllm_plugin.plugin:MyMtllmMachine"

    [build-system]
    requires = ["poetry-core>=1.0.0"]
    build-backend = "poetry.core.masonry.api"
    ```

4.  **Install the plugin:** From the `my-mtllm-plugin/` directory, install the package in development mode.

    ```bash
    pip install -e .
    ```

***

## Usage

Once installed, the plugin will be automatically discovered by Jaclang. You can test it by running a Jaclang file that uses the `by llm()` feature.

1.  Create a file named `test.jac` with the following content:

    ```jaclang
    import:py from mtllm, Model;

    glob llm = Model(model_name="gpt-3.5-turbo");

    can test_plugin {
        result = get_answer("What is 2+2?") by llm();
        print(result);
    }

    can get_answer(question: str) -> str by llm();

    with entry {
        test_plugin();
    }
    ```

2.  Run the Jaclang file from your terminal:

    ```bash
    jac run test.jac
    ```

You will see the output from your plugin's `print` statements, confirming that it successfully intercepted the LLM call before the final result is printed.

***

## Extending the Plugin

This example serves as a template. You can modify the `call_llm` method to implement more advanced features, such as:
- **Logging and Monitoring:** Track call durations, token usage, and errors.
- **Caching:** Store and retrieve responses to avoid redundant API calls.
- **Custom Model Integration:** Use the plugin to connect to a fine-tuned model (like TinyLLaMA) that you have hosted locally or on a different service, making it accessible via Jaclang.
