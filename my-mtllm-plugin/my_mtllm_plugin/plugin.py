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
import requests


class MyMtllmMachine:
    """Custom MTLLM Plugin Implementation."""

    @staticmethod
    @hookimpl
    def call_llm(
        model: Model, caller: Callable, args: dict[str | int, object]
    ) -> object:
        """Custom LLM call implementation."""
        # Custom logic implementation
        # print(f"model name: {model}, caller: {caller}, args: {args}")
        # print(f"Custom plugin intercepted call to: {caller.__name__}")
        # print(f"Arguments: {args}")

        
        # Create the MTIR object using the factory method
        mtir_object = MTIR.factory(
        caller=caller,
        args=args,
        call_params={} 
    )
        
        # Get the return JSON from the MTIR object and print it
        #
        # print(f"MTIR return JSON: {mtir_json}")
        # print("DONE!!!!!")

        which_model = requests.get("https://5wwzgp9vqbnlz3-7000.proxy.runpod.net/which")
        print(f"GET /which response: {which_model.text}")
        # Parse the JSON response and check if is_local is True (as a string)
        try:
            is_local = which_model.json().get("is_local", "False") == "True"
        except Exception as e:
            print(f"Error parsing /which response: {e}")
            is_local = False
        if not is_local:
            print("Model is Global ")
            final_result = model.invoke(mtir_object)


        llm2 = Model(
            model_name="gpt-4o",            # can be any string, required for LiteLLM
            api_key="not-needed",           # dummy, local endpoint doesn’t check
            proxy_url="https://5wwzgp9vqbnlz3-7000.proxy.runpod.net"
        )
        print("Running Thenu")
        result = llm2.invoke(mtir_object)
        print(f"Result: {result}")
        print(f"Type of result: {type(result)}")

        if(is_local):
            final_result = result
        
        print(f"Result from model.invoke: {result}")
        # print(f"Message content from litellm: {message_content}")
        #print(f"Result: {result}")
        return final_result


