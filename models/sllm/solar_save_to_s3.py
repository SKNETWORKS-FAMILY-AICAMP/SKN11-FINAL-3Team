import boto3
import os
from google.colab import userdata

local_dir = "./solar-qlora-4bits"
bucket = userdata.get('AWS_S3_BUCKET_NAME')
prefix = "base-model/solar"

session = boto3.Session(
    aws_access_key_id=userdata.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=userdata.get('AWS_SECRET_ACCESS_KEY'),
    region_name=userdata.get('AWS_S3_REGION')
)
s3 = session.client("s3")

def upload_dir_to_s3(local_dir, bucket, prefix, s3_client):
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            s3_key = os.path.join(prefix, os.path.relpath(local_path, local_dir)).replace("\\", "/")

            s3_client.upload_file(local_path, bucket, s3_key)
            print(f"✅ 업로드 완료: s3://{bucket}/{s3_key}")

upload_dir_to_s3(local_dir, bucket, prefix, s3)