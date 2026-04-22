# DAYS 5-6 - training/finetune.py
# Run this on GPU (Google Colab or Kaggle notebook if no local GPU)
# Command: python training/finetune.py
#
# FREE GPU OPTIONS:
#   1. Google Colab: colab.research.google.com (free T4 GPU, 12hrs)
#   2. Kaggle Notebooks: kaggle.com/code (free T4, 30hrs/week)
#   Upload this file + data/ folder to either platform

import os, torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

print("=" * 60)
print("RuralMED - Fine-tuning Gemma 4 E4B with Unsloth")
print("=" * 60)
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Load base model ───────────────────────────────
print("\n[1/5] Loading Gemma 4 E4B base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = "google/gemma-4-e4b-it",
    max_seq_length= 2048,
    dtype         = None,          # auto-detect
    load_in_4bit  = True,          # 4-bit quantization keeps it small
)
print("Base model loaded!")

# ── Add LoRA adapters ─────────────────────────────
print("\n[2/5] Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r                   = 16,
    target_modules      = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha          = 16,
    lora_dropout        = 0,
    bias                = "none",
    use_gradient_checkpointing = "unsloth",
    random_state        = 42,
    use_rslora          = False,
    loftq_config        = None,
)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}% of total)")

# ── Load dataset ──────────────────────────────────
print("\n[3/5] Loading training dataset...")
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train.jsonl",
        "validation": "data/val.jsonl"
    }
)
print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

# ── Configure training ────────────────────────────
print("\n[4/5] Starting training...")
trainer = SFTTrainer(
    model             = model,
    tokenizer         = tokenizer,
    train_dataset     = dataset["train"],
    eval_dataset      = dataset["validation"],
    dataset_text_field= "text",
    max_seq_length    = 2048,
    dataset_num_proc  = 2,
    packing           = False,
    args = TrainingArguments(
        per_device_train_batch_size   = 2,
        gradient_accumulation_steps   = 4,
        warmup_steps                  = 50,
        num_train_epochs              = 3,
        learning_rate                 = 2e-4,
        fp16                          = not torch.cuda.is_bf16_supported(),
        bf16                          = torch.cuda.is_bf16_supported(),
        logging_steps                 = 25,
        eval_steps                    = 200,
        evaluation_strategy           = "steps",
        save_steps                    = 500,
        save_total_limit              = 2,
        output_dir                    = "training/checkpoints",
        optim                         = "adamw_8bit",
        weight_decay                  = 0.01,
        lr_scheduler_type             = "cosine",
        seed                          = 42,
        report_to                     = "none",
        load_best_model_at_end        = True,
    ),
)

# Show training stats
gpu_stats     = torch.cuda.get_device_properties(0)
start_gpu_mem = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory    = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"Peak reserved memory = {start_gpu_mem} GB.")

trainer_stats = trainer.train()
print(f"\nTraining complete!")
print(f"Time: {trainer_stats.metrics['train_runtime']:.0f}s")
print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")

# ── Save model in GGUF for Ollama ─────────────────
print("\n[5/5] Saving model in GGUF format for Ollama...")
os.makedirs("training/ruralmED-gemma4", exist_ok=True)

model.save_pretrained_gguf(
    "training/ruralmED-gemma4",
    tokenizer,
    quantization_method = "q4_k_m"   # best quality/size balance
)

print("\nModel saved to training/ruralmED-gemma4/")
print("\nNext step: Run this to load into Ollama:")
print("  ollama create ruralmED -f training/Modelfile")
print("\nThen test:")
print('  ollama run ruralmED "Patient has fever 39C for 3 days, headache, no neck stiffness"')
