#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 18:39:01 2025

@author: constantina
"""

import os
from pathlib import Path
import subprocess
import tempfile
import time
from pydub import AudioSegment
from custom_packages.misc_modules import styling_and_animations, config
import pandas as pd


def transcribe_all_videos(video_data: pd.DataFrame) -> str:
    """
    Initiates the transcription process by retrieving the id and url of each 
    video and passing it on to the audio transcription service.

    Parameters
    ----------
    video_data : pd.Dataframe
        Contains json youtube data such as the video url, id, published_date in
        a dataframe format for easier data manipulation.

    Returns
    -------
    str
        The combined text across all transcription files.

    """
    
    remaining_videos_countdown = len(video_data)
    videos_counter = 1
    for row in video_data.itertuples(index=False):
        try:
            styling_and_animations.print_highlighted_message(f"\nAbout to start processing video {row.video_id} ({videos_counter}/{len(video_data)}) for transcription")
            #full_transcription = transcriber.transcribe_youtube_audio_from_videos(row.video_id, row.url)
            combined_transcriptions_text = transcribe_youtube_audio_from_videos("L9YWmR3Gjuk", "https://www.youtube.com/watch?v=L9YWmR3Gjuk")
            if combined_transcriptions_text is None:
                raise RuntimeError(f"Audio processing failed for video {row.video_id} ({row.url})")
            else:
                styling_and_animations.print_highlighted_message(f"\nThis is the completed, full transcription for video {row.video_id} ({row.url}):\n")
                print(combined_transcriptions_text)
            remaining_videos_countdown -= 1
            videos_counter += 1
            styling_and_animations.print_highlighted_message(f"\n{remaining_videos_countdown} videos remaining to be transcribed out of a total of {len(video_data)}")
        except RuntimeError as e:
            styling_and_animations.print_error_message(f"Error ({type(e).__name__}) while saving:", e)
            break
    
    return combined_transcriptions_text


def process_audio_stream(video_url: str) -> str:
    """
    1. Downloading the best audio stream, starting from m4a if available, as its small size
    is convinient for my local resources.
    2. Converting that audio stream to a mono 16kHz mp4 audio file.
    3. Storing that audio file as a temporary resource.

    Parameters
    ----------
    video_url : str
        Used to identify and retrieve the online video from Youtube.

    Returns
    -------
    str
        The path directory of the temporarily stored mp4 audio file.
        If failed, returns empty string.

    """
    
    # Creating a temporary audio file locally for processing, with a random filename.
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_audio_file:
        temp_audio_path = temp_audio_file.name
    
    # Construct yt-dlp command
    yt_dlp_cmd = [
        "yt-dlp",
        "--user-agent", "Mozilla/141.0",
        "--no-playlist",
        "-f", "bestaudio[ext=m4a]/bestaudio",  # fallback to bestaudio if m4a fails
        "-o", "-",                             # output to stdout
        video_url
    ]

    # Construct ffmpeg command to pipe in and resample
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", "pipe:0",
        "-ar", "16000",        # 16kHz
        "-ac", "1",            # mono
        "-c:a", "aac",
        "-b:a", "128k",
        "-f", "mp4",
        temp_audio_path,
        "-y",                  # overwrite
        "-loglevel", "quiet"
    ]
    
    try:
        styling_and_animations.print_animated_ellipsis_message("Starting to download segments of the audio stream")
        yt_process = subprocess.Popen(yt_dlp_cmd, stdout=subprocess.PIPE)
        
        styling_and_animations.print_animated_ellipsis_message("Converting and storing the audio stream while downloading")
        subprocess.run(ffmpeg_cmd, stdin=yt_process.stdout, check=True)
        yt_process.stdout.close()
        yt_process.wait()
        
        return temp_audio_path

    except subprocess.CalledProcessError as e:
        styling_and_animations.print_error_message(f"Error ({type(e).__name__}) during audio download or conversion:", e)
        os.remove(temp_audio_path)
        return None
    except Exception as e:
        styling_and_animations.print_error_message(f"Error ({type(e).__name__}) during audio stream processing:", e)
        os.remove(temp_audio_path)
        return None

    return temp_audio_path


def transcribe_youtube_audio_from_videos(video_id: str, video_url: str, trim_duration_secs = 2400) -> str:
    """
    1. Retrieving the Youtube video based on the URL in an m4a format.
    2. Using OpenAI's whisper-1 model to transcribe all temporary-stored wav audio files.

    Parameters
    ----------
    video_id : str
        Used to mark the transcription file for identification purposes.
    video_url : str
        Used to identify and retrieve the online video from Youtube.

    Returns
    -------
    String
        The full text of the transcription.
        
    """
    
    full_transcribed_text = ""
    
    try:
        temp_audio_path = process_audio_stream(video_url)
        if temp_audio_path is None:
            raise RuntimeError(f"Audio processing failed for video {video_id}")
        
        # OpenAI Whisper has a file size limit of 25mb.
        segmentation_needed = check_file_size_exceeded_limit(temp_audio_path, config.WHISPER_MAX_FILESIZE_MB)
        
        if segmentation_needed:
            styling_and_animations.print_animated_ellipsis_message(f"Preparing to segment the audio of video {video_id}")
            segmentation_success = subset_audio_with_overlap(video_id,temp_audio_path,config.audio_segments_dir)
            
            if segmentation_success:
                styling_and_animations.print_animated_ellipsis_message(f"Preparing to transcribe the segmented audio of video {video_id}")
                full_transcribed_text = transcribe_segmented_audio(video_id, config.audio_segments_dir, str(config.transcript_output_dir))
        
        else:
            styling_and_animations.print_animated_ellipsis_message(f"Preparing to transcribe the audio of video {video_id}")
            transcript_filepath = config.transcript_output_dir / f"{video_id}_full_transcript_youtube.txt"
            full_transcribed_text = transcribe_audio(video_id, temp_audio_path, str(transcript_filepath))
    
    except RuntimeError as e:
        styling_and_animations.print_error_message(f"Error ({type(e).__name__}) while saving:", e)
        return None
    
    finally: # All done by now, so delete the local, temporary audio file.
        if temp_audio_path and Path(temp_audio_path).exists():
            os.remove(temp_audio_path)
    
    return full_transcribed_text



def check_file_size_exceeded_limit(path: str, max_size_limit: float) -> bool:
    """
    Check if the file size in the given path exceeds the expected maximum size limit.

    Parameters
    ----------
    path : str
        The path where the file is located.
    max_size : float
        The maximum size limit permitted for this file.

    Returns
    -------
    bool
        True if the size of the file exceeds the indicated maximum value;
        False if within the maximum size limit.

    """
    
    size_bytes = Path(path).stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print("Current file path:", path)
    print(f"Current file size: {size_mb:.2f} MB")
    
    if size_mb > max_size_limit:
        return True
        
    return False


def subset_audio_with_overlap(audio_id: str, audio_file_path: str, output_dir: str, chunk_length_ms: int = 2400000, overlap_ms: int = 1000) -> bool:
    """
    Segment audio files (with pydub) larger than 25MB into smaller segments, to support OpenAI Whisper's transcription.

    Parameters
    ----------
    video_id : str
        Used to mark the smaller audio file segments for identification purposes.
    audio_file_path : str
        To locate and retrieve the large audio file to be segmented.
    output_dir : str
        Location path to save all smaller audio file segments.
    chunk_length_ms : int (optional)
        How long, in milliseconds, each segment duration should be.
        Default is 2.4m milliseconds (i.e., 40 minutes).
        1 minute = 60k milliseconds.
    overlap_ms : int
        How many milliseconds should the overlap be between segments.

    Returns
    -------
    Boolean
        True if successfully segmented and stored in the output_dir.
        Default is False.
    """
    
    audio = AudioSegment.from_file(audio_file_path)

    os.makedirs(output_dir, exist_ok=True)

    total_length = len(audio) # Duration in milliseconds.
    start = 0 # Indication of starting to segment the main audio file from the beginning.
    audio_counter = 1 

    while start < total_length:
        end = start + chunk_length_ms
        segment = audio[start:end]
        segment = segment.set_frame_rate(16000).set_channels(1)
        
        try:
            output_audio_filepath = Path(output_dir) / f"{audio_id}_segment_{audio_counter:02d}.m4a"
            segment.export(str(output_audio_filepath), format="mp4", bitrate="128k")  # creates m4a container
            styling_and_animations.print_success_message(f"\nSaved: {output_audio_filepath} ({round(len(segment)/60000, 2)} min)")
        except Exception as e:
            styling_and_animations.print_error_message(f"Error ({type(e).__name__}) while saving:", e)
            return False
        
        start = end - overlap_ms
        audio_counter += 1
    
    styling_and_animations.print_highlighted_message(f"\nExported {audio_counter - 1} audio segments in total for video {audio_id}")
    return True



def transcribe_audio(video_id: str, audio_filepath: str, transcript_filepath: str) -> str:
    """
    Retrieves the audio file from the given filepath, transcribes it using the whisper-1 model, 
    and stores it based on the video id as a .txt file in the given filepath. 
    """
    styling_and_animations.print_animated_ellipsis_message(f"Preparing to transcribe the audio of video {video_id}")
    tic = time.perf_counter()
    
    styling_and_animations.print_animated_ellipsis_message(f"Transcribing audio: {Path(audio_filepath).name}")
    try:
        with open(audio_filepath, "rb") as current_audio:
            transcript = config.get_openai_client().audio.transcriptions.create(
                file = current_audio,
                model = "whisper-1"
            )
    except Exception as e:
        styling_and_animations.print_error_message(f"\nError ({type(e).__name__}) transcribing {audio_filepath}:", e)
        return None
        
    toc = time.perf_counter()
    print(f"\n ---------- Transcription lasted for {toc - tic:0.4f} seconds. ----------")
    
    store_transcript(video_id, transcript.text, transcript_filepath)
    
    return transcript.text


def transcribe_segmented_audio(video_id: str, audio_dir: Path, transcript_dir: str) -> str:
    """
    Retrieves the audio segment files from the given filepath based on the video id, transcribes them using 
    the whisper-1 model, and stores them as a concatenated string as a .txt file in the given filepath with 
    the video id argument as part of the transcript name. 
    """
    full_transcript_text = ""
    transcript_segments = []
    audio_counter = 1

    # Filter and sort all chunks for this video. It's crucial that I'm using sorted to transcribe them IN ORDER.
    matching_segments_filepaths = sorted([
        str(filepath) for filepath in audio_dir.iterdir()
        if filepath.name.startswith(video_id) and filepath.name.endswith(".m4a")
    ])
    
    if not matching_segments_filepaths:
        styling_and_animations.print_warning_message(f"\nTranscription failed: No audio segments found for video {video_id} in {audio_dir}")
        return None
    
    tic = time.perf_counter()
    for audio_segment_filepath in matching_segments_filepaths:
        
        transcript_filepath = transcript_dir / f"{video_id}_segment_{audio_counter:02d}_transcript.txt"
        transcript_text = transcribe_audio(video_id, audio_segment_filepath, transcript_filepath)
        transcript_segments.append(transcript_text)
        audio_counter += 1
        
    full_transcript_text = "\n".join(transcript_segments)
    transcript_filepath = transcript_dir / f"{video_id}_full_transcript_youtube.txt"
    store_transcript(video_id, full_transcript_text, transcript_filepath)
        
    toc = time.perf_counter()
    print(f"\n---------- Total transcription time for {video_id}: {toc - tic:.2f} seconds ----------")
    
    return full_transcript_text


def store_transcript(video_id: str, transcript_text: str, transcript_filepath: str) -> bool:
    """
    Stores the transcribed text as a .txt file in the given filepath, using video id as part of the .txt filename.

    Parameters
    ----------
    video_id : str
        The id of the video's transcript to be used as part of the .txt filename.
    transcript_text : str
        The full transcribed text.
    transcript_filepath : str
        The full path to store the .txt transcript.

    Returns
    -------
    bool
        True if successfully stored locally, otherwise False.

    """
    try:
        
        with Path(transcript_filepath).open("w", encoding="utf-8") as transcript_file:
            transcript_file.write(transcript_text.strip())
        styling_and_animations.print_success_message(f"\nSaved transcript for {video_id}: {transcript_filepath}")
    except Exception as e:
        styling_and_animations.print_error_message(f"\nError ({type(e).__name__}) saving transcript for {video_id}:", e)
        return False
    
    time.sleep(0.3)
    
    return True
    