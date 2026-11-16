import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained("models/opt-125m")
# テスト用のデータセットを読み込むWikiText2
test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
test_ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids

# スコアを計測する関数
# 評価のみのため @torch.no_grad()
@torch.no_grad()
def ppl(model, seqlen=2048):
    n = test_ids.shape[1] // seqlen
    nll = 0
    for i in range(n):
        b = test_ids[:, i*seqlen:(i+1)*seqlen].to(model.device)
        nll += model(b, labels=b).loss.float() * seqlen
    return torch.exp(nll / (n * seqlen)).item()

fp16 = AutoModelForCausalLM.from_pretrained(
    "models/opt-125m",
    torch_dtype=torch.float16,
    device_map="auto"
)

gptq = AutoModelForCausalLM.from_pretrained(
    "models/opt-125m-gptq-4bit",
    device_map="auto",
    quantization_config=GPTQConfig(bits=4, backend="triton")
)

result_fp16 = ppl(fp16)
result_gptq = ppl(gptq)

print("FP16:", result_fp16)
print("GPTQ 4bit:", result_gptq)
