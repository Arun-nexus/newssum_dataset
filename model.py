import os

import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from bert_score import score as bert_score
base_model = "microsoft/phi-2"
adapter_path = "./model_trained_parameter"

device = "cpu"

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("hf_token missing")

tok = AutoTokenizer.from_pretrained(
    base_model,
    token=hf_token
)
tok.pad_token = tok.eos_token


mdl = AutoModelForCausalLM.from_pretrained(
    base_model,
    dtype=torch.float32,
    low_cpu_mem_usage=False,
    token=hf_token
)

mdl = PeftModel.from_pretrained(
    mdl,
    adapter_path
)

mdl.to(device)
mdl.eval()

def gen(prompt: str, max_tokens: int = 150) -> str:
    inp = tok(prompt, return_tensors="pt")

    with torch.no_grad():
        out = mdl.generate(
            **inp,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty = 1.2,
            eos_token_id =tok.eos_token_id,
        )

    return tok.decode(out[0], skip_special_tokens=True)
print(gen("what happened during the allahabad stampede?"))

