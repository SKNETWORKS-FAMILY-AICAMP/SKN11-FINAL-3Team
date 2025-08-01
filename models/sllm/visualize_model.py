from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
from torchviz import make_dot

login(token="hf_token_key")

model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer("안녕 나는 유경이야!", return_tensors="pt")

# png로 저장
outputs = model(**inputs)
logits = outputs.logits

dot = make_dot(logits[0, 0], params=dict(model.named_parameters()))
dot.render("image", format="png")

# 모델 계층 프린트
print(model) 
