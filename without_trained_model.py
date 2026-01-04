from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import os

hf_token = os.getenv("HF_TOKEN")

def load_base_phi2():
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2", token=hf_token)
    tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        dtype=torch.float32,
        token=hf_token
    )

    mdl.to("cpu")
    mdl.eval()
    return tok, mdl
def gen_with(tok, mdl, prompt, max_tokens=150):
    inp = tok(prompt, return_tensors="pt")

    with torch.no_grad():
        out = mdl.generate(
            **inp,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=tok.eos_token_id,
        )

    txt = tok.decode(out[0], skip_special_tokens=True)

    if "Answer:" in txt:
        txt = txt.split("Answer:", 1)[1].strip()

    return txt

