files = [
    "physics_converted.jsonl",
    "chemistry_converted.jsonl",
    "computer_converted.jsonl"
]

with open("neb_science_merged.jsonl", "w", encoding="utf-8") as out:
    for fname in files:
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                out.write(line)

print("Merged dataset created: neb_science_merged.jsonl")
