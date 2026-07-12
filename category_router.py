from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import re

# 1. Labeled example prompts (Unchanged)
EXAMPLES = {
    "factual": [
        "What is the capital of France?", "Explain how photosynthesis works.",
        "What causes seasons to change on Earth?", "Define the term 'inflation' in economics.",
        "How does a car engine convert fuel into motion?", "What is the difference between TCP and UDP?",
        "Explain the concept of supply and demand.", "What is quantum entanglement?",
        "How do vaccines work?", "What is the boiling point of water at sea level?",
        "Describe how the immune system fights infection.", "What is the theory of relativity?",
    ],
    "math": [
        "What is 15% of 240?", "If a train travels 60 mph for 3 hours, how far does it go?",
        "A store marks up an item by 20% then discounts it by 10%.", "Calculate total cost of items.",
        "Solve for x: 3x + 7 = 22.", "What is the average of 12, 18, 25, and 31?",
    ],
    "sentiment": [
        "Classify the sentiment of this review.", "Is the following tweet positive or negative?",
        "Label the tone of this customer feedback.", "Analyze the emotional tone of this message.",
    ],
    "summarization": [
        "Summarise the following text in one sentence.", "Condense this article into three bullet points.",
        "Provide a short summary of the passage.", "Write a concise TL;DR.",
    ],
    "ner": [
        "Extract all named entities.", "Identify and label every person, place, and organization.",
        "Find all dates and locations.", "Extract names of people, companies, and cities.",
    ],
    "debugging": [
        "Find the bug in this Python function.", "This code throws an IndexError.",
        "Debug the following JavaScript snippet.", "There is a logic error in this loop.",
    ],
    "codegen": [
        "Write a function that checks if a string is a palindrome.", "Implement a function to reverse a linked list.",
        "Write a Python function that returns the nth Fibonacci number.", "Implement a binary search function.",
    ],
    "logic": [
        "Solve this puzzle.", "Determine the correct seating arrangement.",
        "Who is the tallest?", "Deduce the correct order of the events.",
    ],
}

CATEGORIES = list(EXAMPLES.keys())

# 2. Build the classifier (Unchanged)
class CategoryRouter:
    def __init__(self, n_neighbors: int = 5, model_name: str = "./model_cache/all-MiniLM-L6-v2"):
        print(f"Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        texts, labels = [], []
        for category, examples in EXAMPLES.items():
            texts.extend(examples)
            labels.extend([category] * len(examples))
        embeddings = self.model.encode(texts, show_progress_bar=False)
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
        self.knn.fit(embeddings, labels)
        print("Router ready.\n")

    def classify(self, prompt: str, return_confidence: bool = False):
        emb = self.model.encode([prompt], show_progress_bar=False)
        pred = self.knn.predict(emb)[0]
        if return_confidence:
            distances, indices = self.knn.kneighbors(emb)
            neighbor_labels = [self.knn._y[i] for i in indices[0]]
            confidence = neighbor_labels.count(pred) / len(neighbor_labels)
            return pred, confidence
        return pred

# 3. Regex fallback (UPGRADED to catch Math/Code better)
def regex_override(prompt: str):
    p = prompt.lower()
    # If it looks like code or math, label it early
    if "```" in prompt or any(w in p for w in ["python", "function", "script", "bug", "fix"]):
        return "codegen"
    if any(w in p for w in ["calculate", "solve for", "equation", "formula", "percentage"]):
        return "math"
    if re.search(r"\bsummar(y|ise|ize)\b|\bcondense\b|\btl;dr\b", p):
        return "summarization"
    if "sentiment" in p:
        return "sentiment"
    return None

CONFIDENCE_THRESHOLD = 0.6 

def classify_with_fallback(router: "CategoryRouter", prompt: str):
    override = regex_override(prompt)
    if override:
        return override, "regex_override"
    pred, confidence = router.classify(prompt, return_confidence=True)
    if confidence < CONFIDENCE_THRESHOLD:
        return "factual", "low_confidence_fallback"
    return pred, "knn"

# 4. Model tier mapping (UPGRADED: factual moved to medium)
MODEL_TIER = {
    "sentiment": "easy",     # Local
    "ner": "easy",           # Local
    "summarization": "easy", # Local
    "factual": "medium",     # Fireworks (Moves hard questions to API)
    "math": "medium",        # Fireworks
    "debugging": "hard",     # Fireworks
    "codegen": "hard",       # Fireworks
    "logic": "hard",         # Fireworks
}

# 5. Test harness (Unchanged)
if __name__ == "__main__":
    router = CategoryRouter(n_neighbors=5)
    test_prompts = [
        ("What is the tallest mountain on Earth?", "factual"),
        ("A shirt costs $40 and is on sale for 25% off. What is the sale price?", "math"),
        ("Would you say this hotel review sounds happy or upset?", "sentiment"),
        ("Give me a two-line recap of the following article.", "summarization"),
        ("Pull out the names of companies mentioned in this paragraph.", "ner"),
        ("This function should sort a list but returns it unsorted.", "debugging"),
        ("Write a function that checks whether a number is prime.", "codegen"),
        ("Five students sit in a row. Figure out the seating order.", "logic"),
    ]

    print(f"{'PROMPT':<70} {'PREDICTED':<15} {'SOURCE':<12} {'TIER'}")
    print("-" * 110)
    for prompt, expected in test_prompts:
        predicted, source = classify_with_fallback(router, prompt)
        tier = MODEL_TIER[predicted]
        display_prompt = (prompt[:65] + "...") if len(prompt) > 68 else prompt
        print(f"{display_prompt:<70} {predicted:<15} {source:<12} {tier}")