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

from litellm.types.utils import Message as LiteLLMMessage

from my_mtllm_plugin.utilities import evaluate_local_model

BASE_URL = "https://7184thexftirp2-7000.proxy.runpod.net"


class MyMtllmMachine:
    """Custom MTLLM Plugin Implementation."""

    @staticmethod
    @hookimpl
    def call_llm(
        model: Model, caller: Callable, args: dict[str | int, object]
    ) -> object:
        """Custom LLM call implementation."""
    
        # Create the MTIR object using the factory method
        mtir_object = MTIR.factory(
        caller=caller,
        args=args,
        call_params={} 
    )
        local_llm = Model(
            model_name="gpt-4o",            # can be any string, required for LiteLLM
            api_key="not-needed",           # dummy, local endpoint doesn’t check
            proxy_url=BASE_URL
        )
        # Get the return JSON from the MTIR object and print it
        #
        # print(f"MTIR return JSON: {mtir_json}")
        # print("DONE!!!!!")

        which_model_res = requests.get(BASE_URL+"/which")
        #print(f"GET /which response: {which_model_res.text}")
        # Parse the JSON response and check if is_local is True (as a string)
     
        mode = which_model_res.json().get("mode", "global")
        if mode == "idle":
            #print("Model is Idle ")
            final_result = model.invoke(mtir_object)
            return final_result

        if mode == "global":
            #print("Model is Global ")
            final_result = model.invoke(mtir_object)
            #This collects train data if mode is global
            local_llm.invoke(mtir_object)

        if mode == "local":
            final_result = local_llm.invoke(mtir_object)
        #Eval Mode
        elif (mode =="eval"):
            #print("Model is in Eval ")
            #For Local Usage
            mtir_temp = MTIR.factory(
                caller=caller,
                args=args,
                call_params={}
            )
            result = local_llm.invoke(mtir_temp)
            verdict = evaluate_local_model(model, mtir_temp)
            if verdict:
                #print("Local Model Correct")
                final_result = result
                requests.post(f"{BASE_URL}/eval", json={"verdict": True})
            else:
                #print(f"Local Model Answer: {result}")
                final_result = model.invoke(mtir_object)
                requests.post(f"{BASE_URL}/eval", json={"verdict": False})




        
        return final_result


