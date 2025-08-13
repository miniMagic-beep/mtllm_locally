"""Custom MTLLM Plugin."""

from typing import Callable

from jaclang.runtimelib.machine import hookimpl
from mtllm.llm import Model
from mtllm.mtir import MTIR

import litellm

# This single line is all you need to enable verbose logging
#litellm.set_verbose = True
#litellm.turn_on_debug_logs()
import os


class MyMtllmMachine:
    """Custom MTLLM Plugin Implementation."""

    @staticmethod
    @hookimpl
    def call_llm(
        model: Model, caller: Callable, args: dict[str | int, object]
    ) -> object:
        """Custom LLM call implementation."""
        # Custom logic implementation
        print(f"model name: {model}, caller: {caller}, args: {args}")
        print(f"Custom plugin intercepted call to: {caller.__name__}")
        print(f"Arguments: {args}")

        # Create the MTIR object using the factory method
        mtir_object = MTIR.factory(
        caller=caller,
        args=args,
        call_params={} 
    )
    

        # Option 1: Modify the call and delegate to the original model
        result = model.invoke(mtir_object)
        
        #Mihiran wrote this
        os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
        os.environ["OPENAI_API_KEY"] = "not-needed" # Can be any string for local servers

        response = litellm.completion(
            model="openai/tinyllama.gguf",
            messages=[
            {'role': 'system', 'content': 'This is a task you must complete by returning only the output.\nDo not include explanations, code, or extra text—only the result.\n'},
            {'role': 'user', 'content': f'{caller.__name__}'}
            ],
            temperature=0.7
        )
        

        

        # Print the full response object
        # print("--- Full Response Object ---")
        # print(response)

        # Print just the content of the message
        print("\n--- Assistant's Reply ---")
        message_content = response.choices[0].message.content
        print(message_content)

        # Option 2: Implement completely custom logic
        # result = your_custom_llm_logic(caller, args)
        print(f"Result from model.invoke: {result}")
        print(f"Message content from litellm: {message_content}")
        #print(f"Result: {result}")
        return result


