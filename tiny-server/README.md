# Tiny FastAPI Server — Quick Start

Minimal steps to install dependencies and run the server.





## Install dependencies

    pip install fastapi "uvicorn[standard]" accelerate outlines transformers peft datasets

## Run the API

    uvicorn app:app --host 0.0.0.0 --port 7000 --reload