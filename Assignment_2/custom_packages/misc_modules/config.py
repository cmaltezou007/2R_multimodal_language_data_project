#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 14:42:54 2025

@author: constantina
"""

from pathlib import Path
from openai import OpenAI
from google import genai
from googleapiclient.discovery import build

# useful website for creating ansi colour codes:
# https://jakob-bagterp.github.io/colorist-for-python/ansi-escape-codes/extended-256-colors/#extended-palette
ANSI_BLACK = "\033[30m"
ANSI_LIGHT_RED = "\033[1;31m"
ANSI_LIGHT_GREEN = "\033[1;32m"
ANSI_LIGHT_ORANGE = "\033[38;5;214m"
ANSI_BACKGROUND_LIGHT_YELLOW = "\033[48;5;228m"
ANSI_BACKGROUND_PURPLE_PINK = "\033[48;5;91m"
ANSI_RESET_COLOUR = "\033[0m"

my_base_dir = Path.cwd()
my_base_dir_raw = my_base_dir / "data" / "raw"
my_base_dir_cleaned = my_base_dir / "data" / "processed"
json_dir = my_base_dir_raw / "extracted_json_data"
audio_segments_dir = my_base_dir_raw / "extracted_audio_segments"
transcript_output_dir = my_base_dir_raw / "extracted_transcripts"

DATA_PLATFORMS_EXAMINED = ("youtube") # later, reddit too.

# copy-paste your own API keys here.
OPENAI_API_KEY = "" 
GOOGLE_API_KEY = "" # for Google Gemini
YOUTUBE_API_KEY = "" # for Google's Youtube API v3

WHISPER_MAX_FILESIZE_MB = 25.0 # according to OpenAI

def get_openai_client():
 return OpenAI(api_key=OPENAI_API_KEY)

def get_gemini_client():
    return genai.Client(api_key=GOOGLE_API_KEY)
    
def get_youtube_client():   
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    
    