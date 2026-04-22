# RuralMED AI — Offline Clinical Decision Support

> Gemma 4 E4B fine-tuned model bringing frontier AI to rural clinics — no internet required.

**Tracks:** Health & Sciences · Ollama Special Technology · Main Track  
**Challenge:** Gemma 4 Impact Challenge 2026

---

## The Problem

Over 1 billion people live in areas where health workers make life-or-death decisions with no specialist access, no reliable internet, and no clinical decision support tools. A nurse in rural Punjab, Pakistan or sub-Saharan Africa must diagnose malaria, pneumonia, and severe dehydration using only their training and whatever printed protocols they have on hand.

**RuralMED AI solves this.**

---

## What It Does

A health worker types in patient symptoms and vitals. The system returns in seconds:
- Triage priority (Red / Yellow / Green)
- Top 3 differential diagnoses  
- Specific treatment protocol with medication doses
- Referral decision with clear reasoning
- Sources from WHO/MSF guidelines

**Everything runs locally. No internet. No cloud. No data leaves the device.**

---

## Architecture

```
Health Worker
     │
     ▼
Flask Web UI (localhost:5000)
     │
     ▼
RAG Engine (ChromaDB + all-MiniLM-L6-v2)
     │ retrieves relevant WHO/MSF guidelines
     ▼
Ollama (local inference server)
     │
     ▼
Gemma 4 E4B (fine-tuned on MedQA + PubMedQA with Unsloth)
     │
     ▼
Structured Clinical Response
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed
- 10GB disk space (for model)
- GPU recommended for fine-tuning (not required for inference)

### 1. Install Ollama and pull the model
```bash
# Install Ollama from https://ollama.com
ollama pull gemma4:e4b
```

### 2. Clone and install dependencies
```bash
git clone https://github.com/YOURNAME/ruralmED.git
cd ruralmED
python -m venv ruralmED-env
source ruralmED-env/bin/activate   # Windows: ruralmED-env\Scripts\activate
pip install -r requirements.txt
```

### 3. Build the knowledge base
```bash
python rag/build_knowledge_base.py
```

### 4. (Optional) Fine-tune the model
```bash
# Download training data
python data/download_datasets.py

# Prepare data
python training/prepare_data.py

# Fine-tune (requires GPU)
python training/finetune.py

# Load fine-tuned model into Ollama
ollama create ruralmED -f training/Modelfile
```

### 5. Start the app
```bash
# Terminal 1 - Start Ollama
ollama serve

# Terminal 2 - Start Flask
python app/app.py
```

Open browser: **http://localhost:5000**

**Turn off Wi-Fi. It still works. That's the point.**

---

## Fine-tuning Details

| Parameter | Value |
|---|---|
| Base model | google/gemma-4-e4b-it |
| Training data | MedQA-USMLE (3,000 samples) |
| Method | LoRA (r=16, alpha=16) |
| Framework | Unsloth |
| Quantization | Q4_K_M (GGUF) |
| Serving | Ollama |

### Benchmark Results
| Model | MedQA Accuracy |
|---|---|
| Base Gemma 4 E4B | ~XX% |
| RuralMED (fine-tuned) | ~XX% |
| Improvement | +XX% |

*(Fill in after running `python training/evaluate.py`)*

---

## RAG Knowledge Base

Built from WHO IMAI Primary Care Guidelines and MSF Clinical Guidelines:
- Fever assessment protocol
- Malaria diagnosis and treatment
- Pneumonia classification in children
- Diarrhea and dehydration management
- Malnutrition assessment (MUAC)
- Antenatal care red flags
- Triage (ETAT system)
- Newborn danger signs

---

## Project Structure

```
ruralmED/
├── data/
│   ├── download_datasets.py   # Day 3
│   └── prepare_data.py        # Day 4
├── training/
│   ├── finetune.py            # Days 5-6
│   ├── evaluate.py            # Days 7-8
│   └── Modelfile              # Ollama config
├── rag/
│   ├── build_knowledge_base.py # Days 9-10
│   └── inference.py            # Days 11-12
├── app/
│   ├── app.py                  # Days 13-14
│   └── templates/
│       └── index.html          # Days 15-17
├── requirements.txt
└── README.md
```

---

## Impact

This system can run on a $200 laptop in any rural clinic. No subscription. No connectivity. No recurring cost. One deployment serves hundreds of patients per week indefinitely.

Target deployment regions: Rural Punjab (Pakistan), Northern Nigeria, Rural Kenya, Eastern DRC.

---

## License

Apache 2.0 — free to use, modify, and deploy.
