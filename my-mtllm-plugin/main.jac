import from mtllm.llm {Model}

glob llm = Model(model_name="gemini/gemini-2.0-flash");

obj Student {
    has name: str;
    has age: str;
    has sex: str;
}
def create_a_student(promp: str) -> Student by llm();

def print_student(student: Student) {
    print("Student Name: " + student.name);
    print("Student Age: " + student.age);
    print("Student Sex: " + student.sex);
}

with entry {
    text = "Hi My Name is Mihiran. I am 24 years old and I like Apples. And I am Male";
    student = create_a_student(text);
    print_student(student);
}


