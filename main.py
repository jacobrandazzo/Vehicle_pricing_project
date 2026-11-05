from agent import answer

print("Vehicle price estimator. Describe a vehicle, e.g. '2015 Camry' and ask for a plot or include a target mileage and ask for a price estimate. Type 'quit' to exit.")

history = []  # shared across questions so follow-ups like "what about 120k miles?" work

while True:
    question = input("> ").strip()
    if question.lower() in ("quit", "exit", "done", "stop"):
        break
    if not question:
        continue
    print(answer(question, history, verbose=True))
    print()
