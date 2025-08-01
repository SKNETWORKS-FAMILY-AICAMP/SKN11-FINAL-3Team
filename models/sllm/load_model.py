import boto3
import os
import time
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

start = time.time()
print("⏬ S3에서 모델 다운로드 중...")

session = boto3.Session(
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_S3_REGION')
)
s3 = session.client("s3")

def download_s3_model_dir(bucket, prefix, local_dir):
    paginator = s3.get_paginator('list_objects_v2')
    all_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        all_keys.extend(obj["Key"] for obj in page.get("Contents", []))

    for key in tqdm(all_keys, desc="📦 Downloading model from S3"):
        rel_path = os.path.relpath(key, prefix)
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)

download_s3_model_dir(
    bucket=os.environ.get('AWS_S3_BUCKET_NAME'),
    prefix="base-model/solar/",
    local_dir="./solar-qlora-4bits"
)

print("✅ 다운로드 완료! 모델 로드 시작...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained("./solar-qlora-4bits")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|assistant|>"]})

model = AutoModelForCausalLM.from_pretrained(
    "./solar-qlora-4bits",
    quantization_config=quant_config,
    device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))

end = time.time()
print(f"🚀 전체 실행 시간: {end - start:.2f}초")
