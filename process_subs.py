import os
import json
import subprocess
import re
import cv2
from datetime import timedelta

from elevenlabs.client import ElevenLabs
from elevenlabs import save, Voice, VoiceSettings
import boto3
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()


APIKEY = os.getenv("APIKEY")
ACCESSKEY = os.getenv("ACCESSKEY")
ELEVENLABS_APIKEY = os.getenv("ELEVENLABS_APIKEY")
os.environ['ELEVENLABS_APIKEY'] = ELEVENLABS_APIKEY
os.environ['AWS_ACCESS_KEY_ID'] = ACCESSKEY
os.environ['AWS_SECRET_ACCESS_KEY'] = APIKEY
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
client = ElevenLabs(api_key=ELEVENLABS_APIKEY)


def ultimate_pipeline(json_data, videos_path):
    """
    Runs the entire processing pipeline:
      1. Parses the JSON input (which contains video script and prompts) and splits the video script into sentences.
      2. For each sentence, generates speech audio using AWS Polly.
      3. Measures each audio clip’s duration and writes an SRT subtitle file.
      4. Computes a target duration per “section” (JSON entry) that later is used to slow down video segments.
      5. For each section, looks in videos_path for subfolders (named video_000, video_001, …) that contain an 'output.mp4'.
         It then slows down the video using ffmpeg so that its duration matches the target duration.
      6. Concatenates all processed video pieces.
      7. Combines all generated audio sentence files into one audio file.
      8. Overlays the subtitles onto the concatenated video and merges the new video with the combined audio.
      9. Returns the filename of the finished video.
      
    Parameters:
      - json_data: a JSON string (or JSON structure dumped as a string) with the video prompt and video script.
      - videos_path: the path to the root folder where videos are stored.
                     It is expected that videos are in subfolders named video_000, video_001, etc.,
                     each containing an 'output.mp4' file.
    
    Returns:
      - The filename of the final video (finished.mp4).
    """
    
    # Helper Functions
    
    def format_srt_time(seconds):
        """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
        millisec = int((seconds - int(seconds)) * 1000)
        return f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02},{millisec:03}"
    
    def get_video_duration(file_path):
        """
        Uses ffprobe to get the duration of a video file.
        """
        command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "json",
            file_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    
    def slow_down_video_ffmpeg(input_video_path, output_video_path, target_duration):
        """
        Slows down the video so that its new duration matches target_duration.
        """
        original_duration = get_video_duration(input_video_path)
        slowdown_factor = target_duration / original_duration
        print(f"\nSlowing down video:\n  Input: {input_video_path}\n  Original duration: {original_duration:.2f}s, "
              f"Target: {target_duration:.2f}s, Factor: {slowdown_factor:.2f}")
        command = [
            "ffmpeg",
            "-y",  # Force overwrite
            "-i", input_video_path,
            "-filter:v", f"setpts={slowdown_factor}*PTS",
            output_video_path
        ]

        print("Running command:", " ".join(command))
        subprocess.run(command, check=True)
    
    # 1. Prepare Directories and Parse JSON
    
    # Directory for audio files (and generated subtitles)
    audio_dir = "audio_files"
    os.makedirs(audio_dir, exist_ok=True)
    
    # Parse JSON data (expecting a JSON string)
    data = json_data
    script_sentences = {}
    for index, entry in enumerate(data, start=1):
        script = entry.get("video_script", "")
        if script:
            # Split the script into sentences based on period, trim whitespace, and filter empty strings.
            raw_sentences = re.findall(r'[^.]+(?:\.)?', script)
            sentences = []
            for sentence in raw_sentences:
                sentence = sentence.strip()
                if sentence:
                    if not sentence.endswith('.'):
                        sentence += '.'
                    sentences.append(sentence)
            script_sentences[index] = sentences

    # 2. Generate Audio (Speech) and Build SRT Entries
    
    # Create AWS Polly client (use your own secure credentials management in production)     
    srt_entries = []
    section_durations = {}       # Will store the (extended) target duration for each section
    cumulative_durations = {}    # Cumulative timestamp per sentence (for debugging/logging)
    current_time = 0.0           # Running timestamp (in seconds)
    subtitle_index = 1
    previous_section = None      # To decide on silence duration between sentences
    
    sections_ordered = list(script_sentences.keys())
    first_section = sections_ordered[0]
    last_section = sections_ordered[-1]
    
    print("\nGenerating audio clips and building SRT entries...")
    for section, sentences in script_sentences.items():
        total_duration = 0.0  # Total duration (in seconds) for the current section (without extra extension)
        for idx, sentence in enumerate(sentences):
            # Generate speech audio using AWS Polly
            response = client.text_to_speech.convert(
                text=sentence,
                voice_id="7VqWGAWwo2HMrylfKrcm",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )
            file_path = os.path.join(audio_dir, f"section_{section}_sentence_{idx}.mp3")
            save(response, file_path)
             
            # Load the generated audio and get its duration in seconds
            audio_clip = AudioSegment.from_mp3(file_path)
            duration = len(audio_clip) / 1000.0  # milliseconds -> seconds
            
            # Decide on silence duration: no silence before the very first sentence; otherwise:
            # 0.3 sec if in the same section; 1.3 sec if changing sections.
            if previous_section is None:
                silence_duration = 0.0
            elif previous_section == section:
                silence_duration = 0.3
            else:
                silence_duration = 1.3
            current_time += silence_duration  # account for the silence before this sentence
            
            total_duration += duration
            cumulative_durations[f"Sentence {idx+1} (Section {section})"] = current_time + duration
            
            # Create an SRT entry using the current timestamp and duration
            start_time_str = format_srt_time(current_time)
            end_time_str = format_srt_time(current_time + duration)
            srt_entries.append(f"{subtitle_index}\n{start_time_str} --> {end_time_str}\n{sentence}\n\n")
            
            current_time += duration
            subtitle_index += 1
            previous_section = section
        
        # Extend the section’s total duration by a little extra:
        # For the first and last sections add 1.25 seconds; for others add 1.9 seconds.
        if section == first_section or section == last_section:
            extended_duration = total_duration + 1.25
        else:
            extended_duration = total_duration + 1.9
        section_durations[section] = extended_duration
        print(f"  Section {section}: Extended Duration = {extended_duration:.2f} seconds")
    
    # Write the SRT file
    srt_file_path = os.path.join(audio_dir, "subtitles.srt")
    with open(srt_file_path, "w", encoding="utf-8") as srt_file:
        srt_file.writelines(srt_entries)
    print("SRT file created at:", srt_file_path)
    
    # 3. Process Videos – Slow Down Each Section’s Video
    
    # The processed (slowed down) videos will be stored here.
    new_videos_dir = os.path.join("videos", "new_videos")
    os.makedirs(new_videos_dir, exist_ok=True)
    
    print("\nProcessing videos for each section...")
    for section, target_duration in section_durations.items():
        # Map section 1 -> folder video_000, section 2 -> video_001, etc.
        folder_name = f"video_{section - 1:03d}"
        input_video_path = os.path.join(videos_path, folder_name, "output.mp4")
        if not os.path.exists(input_video_path):
            print(f"Warning: Input video not found: {input_video_path}. Skipping section {section}.")
            continue
        
        output_filename = f"videos_{section:03d}.mp4"
        output_video_path = os.path.join(new_videos_dir, output_filename)
        slow_down_video_ffmpeg(input_video_path, output_video_path, target_duration)
    
    # 4. Concatenate the Processed Video Segments
    
    videos_combined_dir = os.path.join(new_videos_dir, "videos_combined")
    os.makedirs(videos_combined_dir, exist_ok=True)
    combined_video_path = os.path.join(videos_combined_dir, "video.mp4")
    
    # Find all processed video files matching the pattern (e.g., videos_001.mp4, videos_002.mp4, …)
    video_files = sorted(
        [f for f in os.listdir(new_videos_dir) if re.match(r'videos_\d+\.mp4$', f)],
        key=lambda x: int(re.search(r'videos_(\d+)\.mp4$', x).group(1))
    )
    video_files = [os.path.join(new_videos_dir, f) for f in video_files]
    if not video_files:
        raise FileNotFoundError("No slowed-down video files found in the expected format.")
    
    # Open the first video to get video properties
    cap = cv2.VideoCapture(video_files[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(combined_video_path, fourcc, fps, (width, height))
    
    print("\nConcatenating video segments...")
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Processing {video_file} ({frame_count} frames)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
    out.release()
    print("Combined video saved at:", combined_video_path)
    
    # 5. Combine the Audio Clips into a Single Audio Track
    
    # Look for the sentence audio files (they have 'sentence' in their filename)
    audio_sentence_files = sorted(
        [f for f in os.listdir(audio_dir) if f.endswith(".mp3") and "sentence" in f]
    )
    combined_audio = AudioSegment.silent(duration=0)
    previous_section = None
    for file in audio_sentence_files:
        # File naming: section_{section}_sentence_{idx}.mp3
        parts = file.split("_")
        section_str = parts[1]  # This is the section number as a string
        audio_clip = AudioSegment.from_mp3(os.path.join(audio_dir, file))
        if previous_section is None:
            silence_duration = 0
        elif previous_section == section_str:
            silence_duration = 300   # 0.3 seconds of silence (300 ms)
        else:
            silence_duration = 1300  # 1.3 seconds (1300 ms) between different sections
        combined_audio += AudioSegment.silent(duration=silence_duration) + audio_clip
        previous_section = section_str
    
    final_audio_path = os.path.join(audio_dir, "combined_audio.mp3")
    combined_audio.export(final_audio_path, format="mp3")
    print("Combined audio saved at:", final_audio_path)
    
    # 6. Add Subtitles and Merge Audio with Video to Create the Final Output
    
    subtitled_video = "videos/subtitled.mp4"
    finished_video = "videos/final_vid.mp4"
    
    # Use ffmpeg to burn in subtitles onto the combined video
    cmd_subtitles = (
        f'ffmpeg -y -i "{combined_video_path}" '
        f'-vf "subtitles={srt_file_path}:force_style=\'BackColour=&HFF000000,BorderStyle=3\'" '
        f'-c:a copy "{subtitled_video}"'
    )    
    print("\nAdding subtitles with command:\n", cmd_subtitles)
    subprocess.run(cmd_subtitles, shell=True, check=True)
    
    # Merge the subtitled video with the combined audio
    cmd_merge = f'ffmpeg -y -i "{subtitled_video}" -i "{final_audio_path}" -c:v copy -c:a aac -strict experimental "{finished_video}"'
    print("Merging audio and video with command:\n", cmd_merge)
    subprocess.run(cmd_merge, shell=True, check=True)
    
    print("\nFinal video created:", finished_video)
    return finished_video
