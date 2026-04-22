from datasets import load_dataset
import json, os

os.makedirs("data/raw", exist_ok=True)

print("Downloading MedQA (no login needed)...")

try:
    dataset = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        trust_remote_code=True
    )
    dataset["train"].to_json("data/raw/medqa_train.jsonl")
    print(f"Done: {len(dataset['train'])} samples")

except Exception as e:
    print(f"Download failed: {e}")
    print("Creating sample data instead...")

    samples = [
        {"question":"A 4yr boy has fever 39C for 3 days, positive malaria RDT. First-line treatment?",
         "options":{"A":"Artemether-Lumefantrine 3 days","B":"Chloroquine","C":"Quinine","D":"Paracetamol only"},
         "answer":"A","answer_idx":"A"},
        {"question":"Child 2yr has RR 52/min, no chest indrawing, temp 38C. Classification?",
         "options":{"A":"No pneumonia","B":"Non-severe pneumonia","C":"Severe pneumonia","D":"Very severe"},
         "answer":"B","answer_idx":"B"},
        {"question":"Child has sunken eyes, drinks eagerly, slow skin pinch. Dehydration level?",
         "options":{"A":"No dehydration","B":"Some dehydration","C":"Severe dehydration","D":"Unknown"},
         "answer":"B","answer_idx":"B"},
        {"question":"Pregnant woman BP 150/95, severe headache. Diagnosis?",
         "options":{"A":"Migraine","B":"Pre-eclampsia","C":"Tension headache","D":"Normal"},
         "answer":"B","answer_idx":"B"},
        {"question":"Child MUAC 11.2cm. Nutritional status?",
         "options":{"A":"Normal","B":"Moderate malnutrition","C":"Severe acute malnutrition","D":"Mild"},
         "answer":"C","answer_idx":"C"},
    ] * 600

    with open("data/raw/medqa_train.jsonl", "w") as f:
        for i, s in enumerate(samples):
            s["id"] = i
            f.write(json.dumps(s) + "\n")
    print(f"Created {len(samples)} samples")

