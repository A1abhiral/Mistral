import json

QUESTION_KEYS = {"question", "q", "question_text"}
ANSWER_KEYS = {"answer", "ans", "solution"}

def extract_questions(obj):
    results = []

    if isinstance(obj, list):
        for item in obj:
            results.extend(extract_questions(item))

    elif isinstance(obj, dict):
        q = None
        a = None

        for k in QUESTION_KEYS:
            if k in obj:
                q = obj[k]

        for k in ANSWER_KEYS:
            if k in obj:
                a = obj[k]

        if q and a:
            results.append({"question": q, "answer": a})
        else:
            for v in obj.values():
                results.extend(extract_questions(v))

    return results


def convert_file(input_file, output_file, subject_name):
    with open(input_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    questions = extract_questions(raw)

    print(f"{subject_name}: Found {len(questions)} questions")

    with open(output_file, "w", encoding="utf-8") as out:
        for item in questions:
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a Grade 12 NEB {subject_name} teacher."
                    },
                    {
                        "role": "user",
                        "content": f"Subject: {subject_name}\nQuestion: {item['question']}"
                    },
                    {
                        "role": "assistant",
                        "content": item["answer"]
                    }
                ]
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


convert_file("physics.json", "physics_converted.jsonl", "Physics")
convert_file("chemistry.json", "chemistry_converted.jsonl", "Chemistry")
convert_file("computer.json", "computer_converted.jsonl", "Computer Science")
