import from mtllm.llm {Model}
import random;

glob llm = Model(model_name="gemini/gemini-2.0-flash");
#glob llm = Model(model_name="gemini/gemini-1.5-pro");


obj Car{
    has make: str;
    has color: str;
    has fuel_type: str;
}

def generate_random_car() -> Car by llm();
def write_a_small_ad(car: Car) -> str by llm();


with entry {
    count: int =1;
    while count <= 101 {
        random_car = generate_random_car();
        print(random_car);
        ad = write_a_small_ad(random_car);
        print(ad);
        count += 1;
    }
    }

