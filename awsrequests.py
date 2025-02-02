import boto3
import json
import logging
from botocore.exceptions import ClientError
import os
from utils import get_video_model_input
from dotenv import load_dotenv

load_dotenv()


APIKEY = os.getenv("APIKEY")
ACCESSKEY = os.getenv("ACCESSKEY")
ELEVENLABS_APIKEY = os.getenv("ELEVENLABS_APIKEY")
os.environ['ELEVENLABS_APIKEY'] = ELEVENLABS_APIKEY
os.environ['AWS_ACCESS_KEY_ID'] = ACCESSKEY
os.environ['AWS_SECRET_ACCESS_KEY'] = APIKEY
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS configuration
SERVICE_NAME = 'bedrock-runtime'
REGION_NAME = 'us-east-1'
ANTHROPIC_VERSION = 'bedrock-2023-05-31'
MAX_TOKENS = 500
TEMPERATURE = 0.5

def get_content_from_llm(prompt, model_id='us.anthropic.claude-3-5-haiku-20241022-v1:0'):
    """
    Sends a prompt to the specified LLM model and returns the response.

    :param prompt: The text prompt to send to the model.
    :param model_id: The identifier of the model to use.
    :return: The JSON response from the model.
    """
    try:
        # Initialize the Bedrock runtime client
        bedrock_runtime = boto3.client(service_name=SERVICE_NAME, region_name=REGION_NAME)
        
        # Build the JSON request body using the prompt as a user message
        body = json.dumps({
            'anthropic_version': ANTHROPIC_VERSION,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': MAX_TOKENS,
            'temperature': TEMPERATURE,
        })
        
        # Invoke the model
        response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
        response_body = json.loads(response.get('body').read())
        return response_body.get('content')[0].get('text')

    except ClientError as err:
        error_message = err.response["Error"]["Message"]
        logger.error(f"A client error occurred: {error_message}")
        return {"error": error_message}
    


def get_video(video_prompt):
    model_input = get_video_model_input(video_prompt)
    bedrock_runtime = boto3.client(service_name=SERVICE_NAME, region_name=REGION_NAME)
    response = bedrock_runtime.start_async_invoke(
        modelId="amazon.nova-reel-v1:0",
        modelInput=model_input,
        outputDataConfig={
            "s3OutputDataConfig": {
                "s3Uri": "s3://bedrock-video-generation-us-east-1-pgo1k7/"
            }
        }
    )
    return response
