from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
"""01 でダウンロードした facebook/opt-125m を読み込み 4bit 量子化する"""

base_dir = "models/opt-125m"
quant_dir = "models/opt-125m-gptq-4bit"

tok = AutoTokenizer.from_pretrained(base_dir)
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tok)
model = AutoModelForCausalLM.from_pretrained(base_dir, quantization_config=gptq_config, device_map="auto")

model.to("cpu")
model.save_pretrained(quant_dir)
tok.save_pretrained(quant_dir)