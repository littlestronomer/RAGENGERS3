import os
import uuid
import time
import re
import cv2
import logging
import shutil  # For moving and deleting folders
import json    # For saving/loading quiz JSON

from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import your helper functions and modules.
from utils import get_video_model_input, check_job_status, download_s3_prefix
from awsrequests import get_content_from_llm, get_video
from video_script import get_video_script_and_quiz  # Returns (video_script, video_quiz)
from process_subs import ultimate_pipeline  # Your processing function
from dotenv import load_dotenv

load_dotenv()


APIKEY = os.getenv("API_KEY")
ACCESSKEY = os.getenv("ACCESSKEY")
ELEVENLABS_APIKEY = os.getenv("ELEVENLABS_APIKEY")
os.environ['ELEVENLABS_APIKEY'] = ELEVENLABS_APIKEY
os.environ['AWS_ACCESS_KEY_ID'] = ACCESSKEY
os.environ['AWS_SECRET_ACCESS_KEY'] = APIKEY
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
# Set up logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
app = FastAPI()

# Base directory where generated videos are saved.
SAVED_VIDEOS = "saved_videos"
os.makedirs(SAVED_VIDEOS, exist_ok=True)

# Mount folders for serving saved videos and static files.
app.mount("/saved_videos", StaticFiles(directory=SAVED_VIDEOS), name="saved_videos")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global flag to prevent concurrent video generations.
video_generation_in_progress = False

def generate_video(prompt: str, output_dir: str):
    """
    Executes the complete video-generation pipeline:
      1. Creates output directories.
      2. Saves the prompt.
      3. Generates the video script (and quiz) from the prompt.
      4. Saves the quiz (video_quiz) as quiz.json in the output folder.
      5. For each video script entry, it requests a video part and waits until all parts are ready.
      6. Downloads each video part into output_dir/parts.
      7. Moves the downloaded parts so that they appear as video_000, video_001, etc.
      8. Calls ultimate_pipeline(video_script, output_dir) which processes subtitles, slows down segments,
         concatenates videos, and creates the final video (initially at the global folder "videos/final_vid.mp4").
      9. Moves the final video into output_dir and creates a preview image.
      10. Deletes temporary folders ("videos" and "audio_files").
    """
    logging.info("Creating output directory at %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    parts_dir = os.path.join(output_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    
    # Save the prompt for later listing.
    prompt_file = os.path.join(output_dir, "prompt.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)
    logging.info("Saved prompt to %s", prompt_file)
    
    # Generate video script (and quiz) from the prompt.
    logging.info("Generating video script for prompt: %s", prompt)
    video_script, video_quiz = get_video_script_and_quiz(prompt)
    # video_script = video_script[:2]
    
    # Save the quiz into quiz.json in the output folder.
    quiz_file = os.path.join(output_dir, "quiz.json")
    with open(quiz_file, "w", encoding="utf-8") as f:
        json.dump(video_quiz, f)
    logging.info("Saved quiz to %s", quiz_file)
    
    video_responses = []  # Will store details for each video part.
    for index, video_script_item in enumerate(video_script):
        logging.info("Requesting video part %d with prompt: %s", index, video_script_item['video_prompt'])
        video_response = get_video(video_script_item['video_prompt'])
        time.sleep(5)  # Optional delay between requests.
        video_script_item['video_response'] = video_response
        video_script_item['invocation_arn'] = video_response["invocationArn"]
        video_responses.append(video_script_item)
    
    def is_all_completed(responses):
        for resp in responses:
            status = check_job_status(resp['invocation_arn'])
            if status is None:
                return False
        return True

    logging.info("Waiting for all video parts to complete...")
    while not is_all_completed(video_responses):
        logging.info("Not all parts are ready yet. Sleeping for 10 seconds...")
        time.sleep(10)
    logging.info("All video parts have completed processing.")
    
    # Update each video script entry with its final URI.
    for item in video_responses:
        item['uri'] = check_job_status(item['invocation_arn'])
    
    # Download each video part into its own subfolder inside parts_dir.
    for index, item in enumerate(video_responses):
        part_output_dir = os.path.join(parts_dir, f"video_{index:03d}")
        os.makedirs(part_output_dir, exist_ok=True)
        video_uri = item['uri'] + "/video.mp4"
        logging.info("Downloading video part %d from %s to %s", index, video_uri, part_output_dir)
        download_s3_prefix(video_uri, part_output_dir)
    
    # Move the downloaded parts from 'parts' into the output_dir so that they appear as video_000, video_001, etc.
    for folder in os.listdir(parts_dir):
        src = os.path.join(parts_dir, folder)
        dst = os.path.join(output_dir, folder)
        logging.info("Moving folder %s to %s", src, dst)
        shutil.move(src, dst)
    os.rmdir(parts_dir)
    
    # Run the processing pipeline.
    logging.info("Running ultimate_pipeline on video_script in folder: %s", output_dir)
    ultimate_pipeline(video_script, output_dir)
    
    # After processing, the finished video is created at the global location "videos/final_vid.mp4".
    global_final = os.path.join("videos", "final_vid.mp4")
    if os.path.exists(global_final):
        final_video_path = os.path.join(output_dir, "final_vid.mp4")
        shutil.move(global_final, final_video_path)
        logging.info("Moved final video from %s to %s", global_final, final_video_path)
    else:
        logging.error("Final video not found at %s", global_final)
        return
    
    # Create a preview image from the first frame of final_vid.mp4.
    cap = cv2.VideoCapture(final_video_path)
    ret, frame = cap.read()
    if ret:
        preview_path = os.path.join(output_dir, "preview.jpg")
        cv2.imwrite(preview_path, frame)
        logging.info("Preview image saved to %s", preview_path)
    else:
        logging.error("Could not read a frame from final video for preview.")
    cap.release()
    
    # Clean up temporary directories.
    for temp_dir in ["videos", "audio_files"]:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info("Deleted temporary folder '%s'", temp_dir)

def generate_video_wrapper(prompt: str, output_dir: str):
    """
    Wrapper that calls generate_video, logs any exceptions, and resets the global flag.
    """
    global video_generation_in_progress
    try:
        logging.info("Starting video generation for prompt: %s", prompt)
        generate_video(prompt, output_dir)
        logging.info("Video generation completed for prompt: %s", prompt)
    except Exception as e:
        logging.error("Error during video generation: %s", e)
    finally:
        video_generation_in_progress = False
        logging.info("video_generation_in_progress reset to False.")

@app.get("/in_progress")
def in_progress():
    """
    Endpoint to return the current video-generation status.
    Used by the front end (e.g., to update the Generate button state).
    """
    global video_generation_in_progress
    return {"in_progress": video_generation_in_progress}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Home page that displays the form to enter a prompt.
    """
    return templates.TemplateResponse("generate.html", {"request": request})

@app.post("/generate")
async def create_video(
    request: Request,
    prompt: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    Receives a prompt. If no video is currently being generated, it starts the video-generation
    pipeline as a background task. If one is already in progress, it returns an error.
    """
    global video_generation_in_progress
    if video_generation_in_progress:
        return templates.TemplateResponse("generate.html", {
            "request": request,
            "error": "A video is already being generated. Please wait until it finishes."
        })
    
    video_generation_in_progress = True
    unique_id = uuid.uuid4().hex
    output_dir = os.path.join(SAVED_VIDEOS, unique_id)
    logging.info("Initiating background task for video generation in folder %s", output_dir)
    background_tasks.add_task(generate_video_wrapper, prompt, output_dir)
    return RedirectResponse(url="/videos", status_code=303)

@app.get("/videos", response_class=HTMLResponse)
async def list_videos(request: Request):
    """
    Lists all saved videos with their preview images and prompt text.
    """
    video_list = []
    for video_id in os.listdir(SAVED_VIDEOS):
        video_dir = os.path.join(SAVED_VIDEOS, video_id)
        if os.path.isdir(video_dir):
            prompt_file = os.path.join(video_dir, "prompt.txt")
            prompt_text = ""
            if os.path.exists(prompt_file):
                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompt_text = f.read()
            preview_path = None
            if os.path.exists(os.path.join(video_dir, "preview.jpg")):
                preview_path = f"/saved_videos/{video_id}/preview.jpg"
            video_list.append({
                "id": video_id,
                "prompt": prompt_text,
                "preview": preview_path
            })
    return templates.TemplateResponse("videos.html", {"request": request, "videos": video_list})

@app.get("/videos/{video_id}", response_class=HTMLResponse)
async def video_detail(request: Request, video_id: str):
    """
    Displays the processed final video (final_vid.mp4) along with its prompt and quiz.
    The page includes tabs to switch between the Video and the Quiz.
    """
    video_path = f"/saved_videos/{video_id}/final_vid.mp4"
    prompt_file = os.path.join(SAVED_VIDEOS, video_id, "prompt.txt")
    quiz_file = os.path.join(SAVED_VIDEOS, video_id, "quiz.json")
    prompt_text = ""
    quiz = None
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    if os.path.exists(quiz_file):
        with open(quiz_file, "r", encoding="utf-8") as f:
            quiz = json.load(f)
    return templates.TemplateResponse("video_detail.html", {
        "request": request,
        "video_path": video_path,
        "prompt": prompt_text,
        "quiz": quiz
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
