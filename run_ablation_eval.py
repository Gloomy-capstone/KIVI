import warnings
warnings.filterwarnings("ignore")
import torch
import random
import argparse
import re
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from datasets import load_dataset

MODEL_PATH = "/mnt/data/gloomyteam/kivi_clone/models/Llama-2-7b-hf"

random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--group_size", type=int, default=32)
parser.add_argument("--residual_length", type=int, default=128)
parser.add_argument("--num_samples", type=int, default=100)
args = parser.parse_args()

def extract_gold(answer_text):
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", answer_text)
    if not m:
        return None
    return m.group(1).replace(",", "").strip()

def extract_pred(output_text):
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", output_text)
    if m:
        return m.group(1).replace(",", "").strip()
    text = output_text.replace(",", "")
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    return matches[-1].strip() if matches else None

class StopOnNewline(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len):
        self.newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
        self.hash_ids = tokenizer.encode("####", add_special_tokens=False)
        self.prompt_len = prompt_len
        self.seen_hash = False
    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0, self.prompt_len:].tolist()
        if self.hash_ids[0] in generated:
            self.seen_hash = True
        if self.seen_hash and input_ids[0, -1].item() == self.newline_id:
            return True
        return False

config = LlamaConfig.from_pretrained(MODEL_PATH)
config.k_bits = 2
config.v_bits = 2
config.group_size = args.group_size
config.residual_length = args.residual_length
config.use_flash = False

model = LlamaForCausalLM_KIVI.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()
model.eval()

enc = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
dataset = load_dataset("gsm8k", "main")

fewshot = ""
for i in range(5):
    fewshot += "Question: " + dataset["train"][i]["question"] + "\n"
    fewshot += "Answer: " + dataset["train"][i]["answer"] + "\n"

correct = 0
total = args.num_samples

for i in range(total):
    q = dataset["test"][i]["question"]
    gold = dataset["test"][i]["answer"]

    prompt = fewshot + "Question: " + q + "\nAnswer:"
    inputs = enc(prompt, return_tensors="pt").input_ids.cuda()

    stopper = StopOnNewline(enc, inputs.shape[1])
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=256,
            do_sample=False,
            stopping_criteria=StoppingCriteriaList([stopper])
        )

    pred_text = enc.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
    pred = extract_pred(pred_text)
    gold_num = extract_gold(gold)

    ok = (pred is not None and gold_num is not None and pred == gold_num)
    correct += int(ok)
    print(f"[{i+1}/{total}] pred={pred} | gold={gold_num} | {'O' if ok else 'X'}")

acc = 100.0 * correct / total
print(f"\n[RESULT] G={args.group_size}, R={args.residual_length} -> Acc={acc:.2f}%")
