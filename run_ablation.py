import warnings
warnings.filterwarnings("ignore")
import torch
import random
import argparse
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
from datasets import load_dataset

MODEL_PATH = "/mnt/data/gloomyteam/kivi_clone/models/Llama-2-7b-hf"

random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--group_size", type=int, default=32)
parser.add_argument("--residual_length", type=int, default=128)
args = parser.parse_args()

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

q = dataset["test"][0]["question"]
prompt = fewshot + "Question: " + q + "\nAnswer:"
inputs = enc(prompt, return_tensors="pt").input_ids.cuda()

with torch.no_grad():
    output = model.generate(inputs, max_new_tokens=256, do_sample=False)

pred_text = enc.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
print("QUESTION:", q)
print("GOLD:", dataset["test"][0]["answer"])
print("MODEL OUTPUT:", pred_text)
print(f"group_size={args.group_size}, residual_length={args.residual_length}")
