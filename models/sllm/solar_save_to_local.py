import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

print('시작')
start = time.time()

model_name = "upstage/SOLAR-10.7B-v1.0"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|assistant|>"]})

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

model.resize_token_embeddings(len(tokenizer))
model = prepare_model_for_kbit_training(model)

tokenizer.save_pretrained("./solar-qlora-4bits")
model.save_pretrained("./solar-qlora-4bits")

end = time.time()
print(f"실행 시간: {end - start:.2f}초")