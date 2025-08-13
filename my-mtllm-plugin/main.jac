import from mtllm.llm {Model}

glob llm = Model(model_name="gemini/gemini-2.0-flash");


def say_good_morning() -> str by llm();

with entry {
    print(say_good_morning());
}


