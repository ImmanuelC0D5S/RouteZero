"""
KNN-based task category classifier for RouteZero-style routing.

Classifies incoming prompts into one of 8 fixed categories using
sentence embeddings + KNN, entirely local (zero Fireworks tokens).

Run directly to see a test report:
    python category_router.py
"""

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# ---------------------------------------------------------------------------
# 1. Labeled example prompts per category (used to fit the KNN)
# ---------------------------------------------------------------------------

EXAMPLES = {
    "factual": [
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "What causes seasons to change on Earth?",
        "Define the term 'inflation' in economics.",
        "How does a car engine convert fuel into motion?",
        "What is the difference between TCP and UDP?",
        "Explain the concept of supply and demand.",
        "What is quantum entanglement?",
        "How do vaccines work?",
        "What is the boiling point of water at sea level?",
        "Describe how the immune system fights infection.",
        "What is the theory of relativity?",
    ],
    "math": [
        "What is 15% of 240?",
        "If a train travels 60 mph for 3 hours, how far does it go?",
        "A store marks up an item by 20% then discounts it by 10%. What is the final price if it originally cost $50?",
        "Calculate the total cost of 7 items priced at $12.50 each.",
        "If John has 3 times as many apples as Mary, and Mary has 8, how many does John have?",
        "What is the compound interest on $1000 at 5% annually for 3 years?",
        "A recipe serves 4 people and needs 2 cups of flour. How much flour is needed for 10 people?",
        "Solve for x: 3x + 7 = 22.",
        "What is the average of 12, 18, 25, and 31?",
        "If a project takes 5 people 10 days, how many days would it take 2 people?",
        "Calculate the percentage increase from 80 to 100.",
        "How many hours are there in 3.5 days?",
    ],
    "sentiment": [
        "Classify the sentiment of this review: 'The food was cold and the service was terrible.'",
        "Is the following tweet positive, negative, or neutral? 'Just had the best coffee of my life!'",
        "Label the tone of this customer feedback and explain why.",
        "Determine whether this comment expresses positive or negative sentiment: 'I would never buy this again.'",
        "What is the sentiment of this product review, and what phrases support your answer?",
        "Analyze the emotional tone of this message and classify it as positive, negative, or neutral.",
        "Is this restaurant review favorable or unfavorable? Justify your classification.",
        "Classify this social media post's sentiment with a brief explanation.",
    ],
    "summarization": [
        "Summarise the following text in one sentence: ...",
        "Condense this article into three bullet points.",
        "Provide a short summary of the passage below in under 50 words.",
        "Give a one-paragraph summary of the following report.",
        "Reduce this document to its key takeaways in two sentences.",
        "Write a concise TL;DR for the following text.",
        "Summarize the main argument of this essay in a single sentence.",
        "Compress this news article into a short headline-style summary.",
    ],
    "ner": [
        "Extract all named entities (person, organization, location, date) from this text as JSON.",
        "Identify and label every person, place, and organization mentioned in the passage below.",
        "Find all dates and locations mentioned in the following paragraph.",
        "List the named entities in this sentence, tagged by type.",
        "Extract entities from the following text: names of people, companies, and cities.",
        "Identify all proper nouns in this text and classify them as PERSON, ORG, LOCATION, or DATE.",
        "Pull out every organization and location referenced in the article below.",
    ],
    "debugging": [
        "Find the bug in this Python function and provide a corrected version:\n```python\ndef add(a, b):\n    return a - b\n```",
        "This code throws an IndexError. Identify the bug and fix it.",
        "Debug the following JavaScript snippet, it's returning undefined unexpectedly.",
        "There is a logic error in this loop. Identify it and provide a corrected implementation.",
        "This function is supposed to return the max value but doesn't. Fix the bug.",
        "The following code has an off-by-one error. Correct it.",
        "Identify why this recursive function causes infinite recursion and fix it.",
        "This SQL query returns duplicate rows unintentionally. Find and fix the bug.",
    ],
    "codegen": [
        "Write a function that checks if a string is a palindrome.",
        "Implement a function to reverse a linked list.",
        "Write a Python function that returns the nth Fibonacci number.",
        "Implement a binary search function in Python.",
        "Write a function that merges two sorted arrays into one sorted array.",
        "Create a function to validate whether a given string is a valid email address.",
        "Write a function that counts the frequency of each word in a text.",
        "Implement a function to flatten a nested list.",
    ],
    "logic": [
        "Three friends each own a different pet. Given the clues below, determine who owns which pet.",
        "Solve this puzzle: exactly one of the following statements is true. Determine which one.",
        "Given the constraints below, determine the correct seating arrangement for five people.",
        "If A is taller than B, and B is taller than C, but shorter than D, who is the tallest?",
        "Each of the five houses is painted a different color. Using the clues, determine the color of each house.",
        "Only one suspect is telling the truth. Use the statements below to determine who committed the crime.",
        "Determine the schedule that satisfies all of the following constraints simultaneously.",
        "Given these logical conditions, deduce the correct order of the events.",
    ],
}

CATEGORIES = list(EXAMPLES.keys())

# ---------------------------------------------------------------------------
# 2. Build the classifier
# ---------------------------------------------------------------------------

class CategoryRouter:
    def __init__(self, n_neighbors: int = 5, model_name: str = "./model_cache/all-MiniLM-L6-v2"):
        print(f"Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)

        texts, labels = [], []
        for category, examples in EXAMPLES.items():
            texts.extend(examples)
            labels.extend([category] * len(examples))

        print(f"Embedding {len(texts)} labeled examples across {len(CATEGORIES)} categories...")
        embeddings = self.model.encode(texts, show_progress_bar=False)

        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
        self.knn.fit(embeddings, labels)
        print("Router ready.\n")

    def classify(self, prompt: str, return_confidence: bool = False):
        emb = self.model.encode([prompt], show_progress_bar=False)
        pred = self.knn.predict(emb)[0]

        if return_confidence:
            # fraction of the k nearest neighbors that agree with the prediction
            distances, indices = self.knn.kneighbors(emb)
            neighbor_labels = [self.knn._y[i] for i in indices[0]]
            confidence = neighbor_labels.count(pred) / len(neighbor_labels)
            return pred, confidence
        return pred


# ---------------------------------------------------------------------------
# 3. Regex fallback for high-confidence obvious signals (optional override)
# ---------------------------------------------------------------------------

import re

def regex_override(prompt: str):
    p = prompt.lower()
    if "```" in prompt and ("bug" in p or "fix" in p or "error" in p):
        return "debugging"
    if re.search(r"\bsummar(y|ise|ize)\b|\bcondense\b|\btl;dr\b", p):
        return "summarization"
    if "sentiment" in p:
        return "sentiment"
    return None


CONFIDENCE_THRESHOLD = 0.6  # fraction of k nearest neighbors that must agree

def classify_with_fallback(router: "CategoryRouter", prompt: str):
    override = regex_override(prompt)
    if override:
        return override, "regex_override"

    pred, confidence = router.classify(prompt, return_confidence=True)

    if confidence < CONFIDENCE_THRESHOLD:
        return "factual", "low_confidence_fallback"

    return pred, "knn"


# ---------------------------------------------------------------------------
# 4. Model tier mapping (fill in real Fireworks model IDs once published)
# ---------------------------------------------------------------------------

MODEL_TIER = {
    "sentiment": "easy",
    "ner": "easy",
    "summarization": "easy",
    "factual": "easy",
    "math": "medium",
    "debugging": "hard",
    "codegen": "hard",
    "logic": "hard",
}


# ---------------------------------------------------------------------------
# 5. Test harness — run this file directly to see visible results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    router = CategoryRouter(n_neighbors=5)

    # Held-out test prompts (NOT in the training examples, deliberately
    # phrased differently to check generalization to "unseen variants")
    test_prompts = [
        ("What is the tallest mountain on Earth?", "factual"),
        ("A shirt costs $40 and is on sale for 25% off. What is the sale price?", "math"),
        ("Would you say this hotel review sounds happy or upset? 'Room smelled like mildew, never again.'", "sentiment"),
        ("Give me a two-line recap of the following article.", "summarization"),
        ("Pull out the names of any companies and people mentioned in this paragraph.", "ner"),
        ("This function should sort a list but returns it unsorted, what's wrong?", "debugging"),
        ("Write a function that checks whether a number is prime.", "codegen"),
        ("Five students sit in a row. Using the clues, figure out the seating order.", "logic"),
        ("Explain how blockchain achieves consensus.", "factual"),
        ("If a car depreciates 15% per year, what is it worth after 2 years if it started at $20,000?", "math"),
    ]

    correct = 0
    print(f"{'PROMPT':<70} {'PREDICTED':<15} {'EXPECTED':<15} {'SOURCE':<12} {'TIER'}")
    print("-" * 130)

    for prompt, expected in test_prompts:
        predicted, source = classify_with_fallback(router, prompt)
        tier = MODEL_TIER[predicted]
        match = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1
        display_prompt = (prompt[:65] + "...") if len(prompt) > 68 else prompt
        print(f"{display_prompt:<70} {predicted:<15} {expected:<15} {source:<12} {tier}  {match}")

    print("-" * 130)
    print(f"Accuracy: {correct}/{len(test_prompts)} ({100*correct/len(test_prompts):.1f}%)")