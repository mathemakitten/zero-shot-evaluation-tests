test_examples = [
    {
        "prompt": """Question: Which is more likely?
A. Andrew is a scientist and is smart.
B. Andrew is a scientist.
Answer:""",
        "classes": [" A", " B"],
        "answer_index": 1
    },
    {
        "prompt": """Q: Which is more likely?
1. Michael is an accountant.
2. Michael is an accountant and is careful.
A:""",
        "classes": [" 1", " 2"],
        "answer_index": 0
    },
    {
        "prompt": """Q: Which is more likely to be true?
1. Jessica is a teacher.
2. Jessica is a teacher and is patient.
A:""",
        "classes": [" 1", " 2"],
        "answer_index": 0
    }
]