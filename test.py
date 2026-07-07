from routellm.controller import Controller

controller = Controller(
    routers=["bert"],
    weak_model="gpt-3.5-turbo",
    strong_model="gpt-4",
)

# prompt1 = "What is the capital of France?"
# prompt2 = "Explain the entire structure of signal processing from the basics of signals by delving into fourier transforms."

# score1 = controller.routers["causal_llm"].calculate_strong_win_rate(prompt1)
# score2 = controller.routers["causal_llm"].calculate_strong_win_rate(prompt2)

# print(f"Easy prompt score: {score1}\nHarder prompt score: {score2}")
test_prompts = [
    ("Trivial", "What color is the sky?"),
    ("Easy", "What is 15 + 27?"),
    ("Medium", "Explain how a binary search algorithm works."),
    ("Hard", "Derive the time complexity of merge sort and prove it using the master theorem."),
    ("Very hard", "Explain the mathematical foundations of the Fourier transform and derive Parseval's theorem from first principles."),
]

for label, prompt in test_prompts:
    score = controller.routers["bert"].calculate_strong_win_rate(prompt)
    print(f"{label}: {score:.4f} — {prompt[:50]}...")