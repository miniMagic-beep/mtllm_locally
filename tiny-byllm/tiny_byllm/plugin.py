"""Custom MTLLM Plugin."""

from typing import Callable
from jaclang.runtimelib.machine import hookimpl
from byllm.llm import Model
from byllm.mtir import MTIR
import copy


import litellm


# This single line is all you need to enable verbose logging
#litellm.set_verbose = True
#litellm.turn_on_debug_logs()
import os
import requests

from litellm.types.utils import Message as LiteLLMMessage

from tiny_byllm.utilities import*

BASE_URL = os.getenv("BASE_URL", "http://localhost:7000")

class TinyMain:
    """Custom MTLLM Plugin Implementation."""

    @staticmethod
    @hookimpl
    def call_llm(model: Model, mtir: MTIR) -> object:
        mtir_object = mtir
        """Custom LLM call implementation."""
        log.debug(f"call_llm called with model: {model}, mtir: {mtir_object}")

        local_llm = Model(
            model_name="gpt-4o",            # can be any string, required for LiteLLM
            api_key="not-needed",           # dummy, local endpoint doesn’t check
            proxy_url=BASE_URL
        )

        try:
            which_model_res = requests.get(BASE_URL+"/which")
        except:
            log.error("Could not connect to the local server. Is it running?")
            final_result = model.invoke(mtir_object)
            return final_result
    

        mode = which_model_res.json().get("mode", "global")
        if mode == "idle":
            log.info("Mode is Idle")
            final_result = model.invoke(mtir_object)
            return final_result

        if mode == "global":
            log.info("Mode is Global")
            final_result = model.invoke(mtir_object)
            #This collects train data if mode is global
            local_llm.invoke(mtir_object)

        if mode == "local":
            log.info("Mode is Local")
            final_result = local_llm.invoke(mtir_object)
        
        #Eval Mode
        elif (mode =="eval"):
            log.info("Modes is Eval")
            #For Local Usage - We need copy because the local answer could be wrong
            mtir_temp = copy.deepcopy(mtir_object)
            result = local_llm.invoke(mtir_temp)
            verdict = evaluate_local_model(model, mtir_temp)
            if verdict:
                log.info("Local Model Correct")
                final_result = result
                requests.post(f"{BASE_URL}/eval", json={"verdict": True})
            else:
                log.info("Local Model Incorrect")
                final_result = model.invoke(mtir_object)
                requests.post(f"{BASE_URL}/eval", json={"verdict": False})

        return final_result


