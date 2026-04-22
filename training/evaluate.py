# DAYS 7-8 - training/evaluate.py
# Run: python training/evaluate.py
# Compares base Gemma 4 E4B vs your fine-tuned RuralMED model
# SAVE THE OUTPUT — you need these numbers for your Kaggle writeup

import json, requests, time, os
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_model(model_name: str, prompt: str, timeout: int = 60) -> str:
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300}
        }, timeout=timeout)
        return resp.json().get("response", "")
    except Exception as e:
        return f"ERROR: {e}"

def extract_answer(text: str) -> str:
    """Extract A/B/C/D answer from model response"""
    text = text.upper()
    for prefix in ["ANSWER: ", "RECOMMENDED ACTION: ", "CORRECT ANSWER: "]:
        if prefix in text:
            idx = text.index(prefix) + len(prefix)
            return text[idx:idx+1]
    # scan for standalone A/B/C/D
    for char in ["A", "B", "C", "D"]:
        if f" {char}." in text or f"\n{char}." in text or f"{char})" in text:
            return char
    return ""

# ── Load validation set ───────────────────────────
print("Loading validation data...")
val_data = []
with open("data/val.jsonl") as f:
    for line in f:
        val_data.append(json.loads(line))

# Use first 100 for speed
test_set = val_data[:100]
print(f"Evaluating on {len(test_set)} examples")

# ── Which models to test ──────────────────────────
# Change "ruralmED" to your model name if different
MODELS = ["gemma4:e4b", "ruralmED"]

results = {m: {"correct": 0, "total": 0, "errors": 0} for m in MODELS}

print("\n" + "=" * 60)
print("RUNNING BENCHMARK — this takes ~20 minutes")
print("=" * 60)

for i, ex in enumerate(tqdm(test_set, desc="Evaluating")):
    text = ex["text"]

    # Extract question portion (before model turn)
    if "<start_of_turn>model" in text:
        question_part = text.split("<start_of_turn>model")[0]
        # Strip system prefix for cleaner prompt
        if "[SYSTEM]:" in question_part:
            question_part = question_part.split("\n\n", 1)[-1]
    else:
        continue

    # Extract correct answer from the formatted response
    correct_answer = ""
    if "RECOMMENDED ACTION: " in text:
        part = text.split("RECOMMENDED ACTION: ")[1]
        correct_answer = part[0]
    elif "ANSWER: " in text:
        part = text.split("ANSWER: ")[1]
        correct_answer = part[0]

    if not correct_answer:
        continue

    for model_name in MODELS:
        response = ask_model(model_name, question_part)
        if "ERROR" in response:
            results[model_name]["errors"] += 1
            continue

        predicted = extract_answer(response)
        results[model_name]["total"] += 1
        if predicted == correct_answer:
            results[model_name]["correct"] += 1

# ── Print results ─────────────────────────────────
print("\n" + "=" * 60)
print("BENCHMARK RESULTS — COPY THESE FOR YOUR WRITEUP")
print("=" * 60)

for model_name in MODELS:
    r = results[model_name]
    if r["total"] > 0:
        acc = 100 * r["correct"] / r["total"]
        print(f"\n{model_name}:")
        print(f"  Correct:  {r['correct']}/{r['total']}")
        print(f"  Accuracy: {acc:.1f}%")
        print(f"  Errors:   {r['errors']}")

if all(results[m]["total"] > 0 for m in MODELS):
    base_acc = 100 * results["gemma4:e4b"]["correct"] / results["gemma4:e4b"]["total"]
    tuned_acc = 100 * results["ruralmED"]["correct"] / results["ruralmED"]["total"]
    improvement = tuned_acc - base_acc
    print(f"\nIMPROVEMENT: +{improvement:.1f}% accuracy from fine-tuning")
    print("\nPaste these numbers in your Kaggle writeup Section 3.")

# ── Save results to file ──────────────────────────
os.makedirs("training", exist_ok=True)
with open("training/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to training/benchmark_results.json")
