#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:53:24 2025

@author: constantina
"""

from pathlib import Path
import traceback
import mimetypes
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from PIL import Image
import shutil
import pytesseract as pt
import pdf2image
import random
import speech_recognition as sr
from jiwer import wer
import ollama
import tempfile
import api_config as api_keys


def print_task_heading(message: str) -> None:
    """
    Shows a heading for each exercise task from 2R Assignment 1, to separate tasks visually.
    This function wasn't part of the assignment, but I wanted a simple but more visually distinct division when
    displaying info/results in the console per task.
    """
    print("\n" + "=" * len(message))
    print(message)
    print("=" * len(message) + "\n")
    

def copy_files_and_export_metadata(input_filepaths: list[Path], ouput_file: Path) -> None:
    """
    1. Retrieves and stores the metadata of each input file (file name, type, size in MBs, 
    file length/duration in seconds if audio or video, image dimensions if an image, 
    total word count if text file) in a .txt file.
    2. It also copies each file into a relevant file type folder.
    
    Parameters
    ----------
    input_filepaths : list[Path]
        A list of all the filepaths of the documents that this function is expected to copy and retrieve metadata from.
    ouput_file : Path
        Full filepath to the .txt file.

    Returns
    -------
    None

    """
    info_list = []
    for file_path in input_filepaths:
        file_info_dict = {}
        try:
            file_info_dict["file_name"] = file_path.stem
            file_info_dict["file_type"] = mimetypes.guess_type(file_path.name)[0]
            file_info_dict["file_size_in_mb"] = round(file_path.stat().st_size / (1024 * 1024), 2)
            
            if "text" in file_info_dict["file_type"]:
                file_info_dict["total_word_count"] = len(file_path.read_text(encoding="utf-8").split())
                copy_file(file_path, ouput_file.parent / "text")
            elif "audio" in file_info_dict["file_type"]:
                audio = AudioSegment.from_file(file_path)
                file_info_dict["duration_in_seconds"] = round(len(audio) / 1000, 2)
                copy_file(file_path, ouput_file.parent / "audio")
            elif "video" in file_info_dict["file_type"]:
                video = VideoFileClip(str(file_path))
                file_info_dict["duration_in_seconds"] = video.duration
                copy_file(file_path, ouput_file.parent / "video")
            elif "image" in file_info_dict["file_type"]:
                with Image.open(file_path) as img:
                    file_info_dict["image_width"] = img.size[0]
                    file_info_dict["image_height"] = img.size[1]
                copy_file(file_path, ouput_file.parent / "images")
            
            info_list.append(file_info_dict)
        except Exception as e:
            print(f"Failed to process {file_info_dict['file_name']}: {str(e)}")
            traceback.print_exc()
            
    with ouput_file.open("w", encoding="utf-8") as file:
        for index, file_info in enumerate(info_list):
            file.write(f"\n========= File {index+1} info =========\n\n")
            for key, value in file_info.items():
                file.write(f"{key}: {value}\n")
                

def copy_file(source_file: Path, destination_folder: Path) -> None:
    """ 
    Simply copies the source file to the destination folder, if the folder 
    already exists, otherwise it first creates it. 
    """
    destination_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(source_file), str(destination_folder / source_file.name))

def create_zip_if_folder_exceeds_mb_limit(folder: Path, size_limit_mb: float = 1.00) -> None:
    """Simply compresses the given folder as a zip file, if it exceeds the given size limit in Mbs. """
    folder_size_mb = round((sum(file.stat().st_size for file in folder.iterdir() if file.is_file())) / (1024 * 1024), 2)
    print(f"Folder '{folder.name}' has a total size of {folder_size_mb} MB.")
    if folder_size_mb > size_limit_mb:
        shutil.make_archive(str(folder), "zip", str(folder.parent), folder.name)
            



def export_text_from_image_pdfs(input_filepaths: list[Path]) -> None:
    """
    Going through each PDF document provided and doing the following steps:
    1. Converting each page in the PDF file into an image to be able to be processed by Tesseract OCR.
    2. Going through each image-page and using Tesseract OCR to extract plain text from it and store it as a .txt file.
    3. Exporting three random sentences from each .txt file for validation purposes.

    Parameters
    ----------
    input_filepaths : list[Path]
        A list of all the relevant filepaths.

    Returns
    -------
    None

    """
    for file_path in input_filepaths:
        try:
            img_pages = pdf2image.convert_from_path(str(file_path))
            output_filepath = file_path.with_name(file_path.stem + "_OCR_transcription.txt")
            with output_filepath.open("w", encoding="utf-8") as file:
                for index, page in enumerate(img_pages):
                    file.write(f"\n========= Page {index+1} =========\n")
                    file.write(pt.image_to_string(page).strip())
            
            export_three_random_sentences_for_validation(output_filepath)
            
        except Exception as e:
            print(f"Failed to process {file_path.name}: {str(e)}")
            traceback.print_exc()


def export_text_from_image_pdfs_with_LLM(input_filepaths: list[Path]) -> None:
    """
    Has pretty much the same purpose as the export_text_from_image_pdfs function, 
    but instead of using Tesseract OCR, it uses the llama3.2-vision LLM.
    
    Parameters
    ----------
    input_filepaths : list[Path]
        A list of all the relevant filepaths.

    Returns
    -------
    None

    """
    for file_path in input_filepaths:
        try:
            img_pages = pdf2image.convert_from_path(str(file_path))
            output_filepath = file_path.with_name(file_path.stem + "_LLM_OCR_transcription.txt")
            
            with output_filepath.open("w", encoding="utf-8") as file:
                for index, page in enumerate(img_pages):
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                        page.save(tmp.name)
                        response = ollama.chat(
                            model="llama3.2-vision",
                            messages=[{
                                "role": "user",
                                "content": "Extract the full text from this image, do not include anything else in the output.",
                                "images": [tmp.name]
                            }]
                        )
                        file.write(f"\n========= Page {index+1} =========\n")
                        file.write(response["message"]["content"].strip())
            
            export_three_random_sentences_for_validation(output_filepath, True)

        except Exception as e:
             print(f"Failed to process {file_path.name}: {str(e)}")
             traceback.print_exc()


def export_three_random_sentences_for_validation(filepath: Path, use_llm: bool = False) -> None:
    """Simply exports three random sentences from the given .txt file, and stores them into a new .txt file. """
    try:
        text = filepath.read_text(encoding="utf-8")
        list_of_sentences = text.split(".")
        sample_sentences = random.sample(list_of_sentences, 3)
        
        if use_llm:
            filename = "_LLM_OCR_validation_sample.txt"
        else:
            filename = "_OCR_validation_sample.txt"
        with filepath.with_name(filepath.stem + filename).open("w", encoding="utf-8") as file:
            for text_sentence in sample_sentences:
                file.write(f"{text_sentence}\n\n")
        
    except Exception as e:
        print(f"Failed to extract random sentences from {filepath.name}: {str(e)}")
        traceback.print_exc()


def transcribe_audio(input_filepaths: list[Path]) -> None:
    """
    Transcribing the mp3 files using CMU Sphinx, and then calculating the accuracy 
    (word-error-rate score) between this transcription and the true transcript.
    CMU Sphinx accepts an sr.AudioData input, hence the initial manipulation of the mp3 file
    to retrieve its AudioData version.
    It stores the transcription in a .txt file for easier access if needed to re-evaluate it 
    or something, without having to repeat the transcription process.

    Parameters
    ----------
    input_filepaths : list[Path]
        A list of all the .mp3 audiofile paths.

    Returns
    -------
    None
        DESCRIPTION.

    """
    for filepath in input_filepaths:
        try:
            audio_data = get_mp3_as_audiodata(filepath)
            speech_recogniser = sr.Recognizer()
            text = speech_recogniser.recognize_sphinx(audio_data) 
            print("\nASR transcribed text:\n", text)
            
            with filepath.with_name(filepath.stem + "_audio_transcription.txt").open("w", encoding="utf-8") as file:
                file.write(text)
            
            true_transcript = filepath.with_suffix(".txt").read_text(encoding="utf-8")
            calculate_word_error_rate(text, true_transcript)
        
        except Exception as e:
            print(f"Failed to transcribe audio {filepath.name}: {str(e)}")
            traceback.print_exc()
   
    
def transcribe_audio_with_LLM(input_filepaths: list[Path]) -> None:
    """
    Has pretty much the same purpose as the transcribe_audio function, 
    but instead of using CMU Sphinx, it uses the gpt-4o-transcribe LLM.
    Similarly, it stores the transcription in a .txt file to not have to use the LLM multiple times
    once it has worked, to test the text, etc - don't want to be charged unnecessarily.

    Parameters
    ----------
    filepath : Path
        A list of all the .mp3 audiofile paths.

    Returns
    -------
    None
        DESCRIPTION.

    """
    for filepath in input_filepaths:
        try:
            full_transcription_text = ""
            audio = AudioSegment.from_file(filepath)
            audio_total_duration_seconds = round(len(audio) / 1000, 2)
            # Apparently gpt-4o-transcribe throws an error if the audio duration is longer
            # than 1400 seconds. So I need to check and shorten them wherever applicable.
            if audio_total_duration_seconds > 1400.00:
                segmented_audio_filepaths = []
                segmented_audio_filepaths = subset_audio_with_overlap(filepath, filepath.parent / "audio_segments")
                for segment_filepath in segmented_audio_filepaths:
                    transcription = api_keys.get_openai_client().audio.transcriptions.create(
                        model = "gpt-4o-transcribe",
                        file = segment_filepath
                    )
                    full_transcription_text += transcription.text + "\n"
            else:
                transcription = api_keys.get_openai_client().audio.transcriptions.create(
                    model = "gpt-4o-transcribe",
                    file = filepath
                )
                full_transcription_text = transcription.text
                print("\nASR transcribed text:\n", full_transcription_text)
            
            with filepath.with_name(filepath.stem + "_LLM_audio_transcription.txt").open("w", encoding="utf-8") as file:
                file.write(full_transcription_text)
            
            true_transcript = filepath.with_suffix(".txt").read_text(encoding="utf-8")
            calculate_word_error_rate(full_transcription_text, true_transcript)
        
        except Exception as e:
            print(f"Failed to transcribe audio {filepath.name}: {str(e)}")
            traceback.print_exc()
        

def get_mp3_as_audiodata(filepath: Path) -> sr.AudioData:
    """Returning the given mp3 file as an AudioData object with a mono channel of 16kHz."""
    sound = AudioSegment.from_mp3(str(filepath))
    sound = sound.set_channels(1).set_frame_rate(16000)
    
    audio_data = sr.AudioData(sound.raw_data, sound.frame_rate, sound.sample_width)
    
    return audio_data    

    
def calculate_word_error_rate(hypothesis: str, reference: str) -> None:    
    """Displaying the word-error-rate and accuracy scores between the current transcript (hypothesis) 
    and the true transcript (reference). """
    error = wer(reference, hypothesis)
    
    print("\nWord-error-rate (WER, %):", round(error, 2))
    print("\nTranscription accuracy (%):", round(1 - error, 2))
    


def subset_audio_with_overlap(audio_file_path: Path, output_dir: Path, segment_duration_ms: int = 1350000, overlap_ms: int = 1000) -> list[Path]:
    """
    Segment given audio file (with pydub) into smaller segments, to support the transcription of the OpenAI gpt-4o-transcribe model.

    Parameters
    ----------
    audio_file_path : Path
        The full path of the mp3 audio file to be segmented.
    output_dir : Path
        Where to store the mp3 segments.
    segment_duration_ms : int, optional
        How long, in milliseconds, each segment duration should be.
        The default is 1350000, to be on the safe side.
    overlap_ms : int, optional
        How many milliseconds should the overlap be between segments. 
        The default is 1000.

    Returns
    -------
    bool
        True if successfully segmented and stored in the output_dir.
        Default is False.

    """
    
    audio = AudioSegment.from_file(audio_file_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_length = len(audio) # Duration in milliseconds.
    start = 0 # Indication of starting to segment the main audio file from the beginning.
    audio_counter = 1 

    while start < total_length:
        end = start + segment_duration_ms
        segment = audio[start:end]
        segment = segment.set_frame_rate(16000).set_channels(1)
        
        try:
            output_audio_filepath = output_dir / f"{audio_file_path.stem}_segment_{audio_counter:02d}.mp3"
            segment.export(str(output_audio_filepath), format="mp3", bitrate="128k")  # creates m4a container
            print(f"\nSaved: {output_audio_filepath} ({round(len(segment)/60000, 2)} min)")
        except Exception as e:
            print(f"Error ({type(e).__name__}) while saving:", e)
            return None
        
        start = end - overlap_ms
        audio_counter += 1
    
    print(f"\nExported {audio_counter - 1} audio segments in total for audio {audio_file_path.stem}")
    return list(output_dir.glob("*.mp3"))
    
    
    
    
    