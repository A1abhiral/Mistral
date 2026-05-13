import random

with open("neb_science_merged.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

split = int(0.9 * len(lines))

with open("train.jsonl", "w", encoding="utf-8") as f:
    f.writelines(lines[:split])

with open("val.jsonl", "w", encoding="utf-8") as f:
    f.writelines(lines[split:])

print("Train/Validation split done.")
