import from mtllm.llm {Model}

glob llm = Model(model_name="gemini/gemini-2.0-flash");

obj Student {
    has name: str;
    has age: str;
}
def create_a_student(prompt: str) -> Student by llm();

def print_student(student: Student) {
    print("Student Name: " + student.name);
    print("Student Age: " + student.age);
}

with entry {
    text = "Hi My Name is Mihiran. I am 24 years old and I like Apples";
    student = create_a_student(text);
    print_student(student);
}


