import random
import boto3
import os

SERVICE_NAME = 'bedrock-runtime'
REGION_NAME = 'us-east-1'
from dotenv import load_dotenv

load_dotenv()


APIKEY = os.getenv("APIKEY")
ACCESSKEY = os.getenv("ACCESSKEY")
ELEVENLABS_APIKEY = os.getenv("ELEVENLABS_APIKEY")
os.environ['ELEVENLABS_APIKEY'] = ELEVENLABS_APIKEY
os.environ['AWS_ACCESS_KEY_ID'] = ACCESSKEY
os.environ['AWS_SECRET_ACCESS_KEY'] = APIKEY
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

bedrock_runtime = boto3.client(service_name=SERVICE_NAME, region_name=REGION_NAME)
def get_video_model_input(video_prompt, fps = 24, duration = 6, dimension = "1280x720"):
    model_input = {
        "taskType": "TEXT_VIDEO",
        "textToVideoParams": {
            "text": video_prompt,
        },
        "videoGenerationConfig": {
            "durationSeconds": duration,
            "fps": fps,
            "dimension": dimension,
            "seed": random.randint(0, 1000000)
        }
    }
    return model_input

def check_job_status(invocation_arn):
    response = bedrock_runtime.get_async_invoke(invocationArn=invocation_arn)
    status = response["status"]
    # print(f"Current Status: {status}")

    if status == "Completed":
        output_location = response["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
        return output_location

    elif status == "Failed":
        print("Job failed.")
        return None
    else:
        return None
    
    
def download_s3_prefix(s3_name, local_dir):
    bucket_name = s3_name.replace("s3://", "").split("/")[0]
    prefix = "/".join(s3_name.replace("s3://", "").split("/")[1:]).split("/")[0]
    print(bucket_name, prefix)
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        # Check if any objects are returned
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # Determine local file path
                # This will recreate the folder structure under local_dir
                relative_path = os.path.relpath(key, prefix)
                local_file_path = os.path.join(local_dir, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                if key.endswith('.mp4'):
                    print(f"Downloading {key} to {local_file_path}...")
                    # Download if it has mp4 in the key
                    s3.download_file(bucket_name, key, local_file_path)
        else:
            print("No objects found with that prefix.")
