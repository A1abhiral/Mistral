import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Retrieval.retrieve import retrieve_book_content

results = retrieve_book_content("What is oscillation?", k=1)

for r in results:
    print("\n📘", r["source"], "Page", r["page"])
    print(r["text"])
