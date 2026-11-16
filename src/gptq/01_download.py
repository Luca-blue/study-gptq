from transformers import AutoModelForCausalLM, AutoTokenizer
""" HF から facebook/opt-125m のモデルとトークナイザをダウンロードしてローカルに保存する"""

AutoModelForCausalLM.from_pretrained("facebook/opt-125m").save_pretrained("models/opt-125m")
AutoTokenizer.from_pretrained("facebook/opt-125m").save_pretrained("models/opt-125m")