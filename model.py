import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "microsoft/phi-2"
ADAPTER_PATH = r"C:\Users\Arun\PycharmProjects\JupyterProject\model_trained_parameter"
OFFLOAD_DIR = "./offload"

os.makedirs(OFFLOAD_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# 1️⃣ Load base model WITH offload
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    dtype=torch.float16,
    offload_folder=OFFLOAD_DIR,
)

# 2️⃣ Load LoRA adapter WITH SAME offload_dir
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    device_map="auto",          # 🔥 REQUIRED
    offload_folder=OFFLOAD_DIR  # 🔥 REQUIRED
)

model.eval()

prompt = "[YEAR=2013]\nWhat happened during the Allahabad stampede?"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        temperature=0.0
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
print("Model device:", next(model.parameters()).device)