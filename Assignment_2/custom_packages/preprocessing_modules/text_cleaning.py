#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 14:11:31 2025

@author: constantina
"""

import pandas as pd
from pathlib import Path
from custom_packages.misc_modules import styling_and_animations
import uuid
import contractions

#nltk.download('punkt')
#nltk.download('stopwords')

def clean_transcripts(filepaths_per_platform: dict[str, list[Path]]) -> pd.DataFrame:
    """
    Cleans transcripts per platform and returns a dataframe where each row contains 
    a unique transcript identifier, the actual filename retrieved from Path, 
    the platform name, the raw text, and the cleaned text.

    Parameters
    ----------
    filepaths_per_platform : dict[str, list[Path]]
        Contains the platform name (e.g., youtube) as key, and the full paths
        to the relevant transcripts as the value.

    Returns
    -------
    pd.DataFrame

    """
    
    transcript_records = []
    for platform_name, filepaths_list in filepaths_per_platform.items():
        for file_path in filepaths_list:
            try:
                with file_path.open("r", encoding="utf-8") as file:
                    raw_text = file.read()
                    cleaned_text = clean_text(raw_text)
                    transcript_records.append({
                        "transcript_id": str(uuid.uuid4()),
                        "filename": file_path.name,
                        "platform": platform_name,
                        "raw_text": raw_text,
                        "cleaned_text": cleaned_text
                    })
            except Exception as e:
                styling_and_animations.print_error_message(f"Failed to get the content from transcript {file_path.name} (Error: {type(e).__name__}):", e)
                
    return pd.DataFrame(transcript_records)


def clean_text(transcript_text: str) -> str:
    """
    Preprocesses the string argument, by stripping unecessary white spaces, 
    converts to lowercase, decouples contractions and replaces dashes with an empty space.

    Parameters
    ----------
    transcript_text : str
        The text to be preprocessed.

    Returns
    -------
    str
        The cleaned text after preprocessing.

    """
    cleaned_text = transcript_text.strip()
    cleaned_text = cleaned_text.lower()
    
    # Fixing the apostrophe issue I'm having - ideally I would have used the 
    # pycontractions library instead as it handles ambiguity better such as 
    # "ain't" (am not / are not / has not, ...), but having some Java versioning 
    # issues at the moment.
    cleaned_text = contractions.fix(cleaned_text) 
    
    # focusing on hyphen removal to avoid issues I 
    # came across such as "not-so-serious" becoming "notsoserious".
    cleaned_text = cleaned_text.replace("-", " ") 
    
    return cleaned_text
    

def store_cleaned_data_as_csv(transcript_records: pd.DataFrame, cleaned_data_dir: Path, raw_data_dir: Path) -> None:
    """Extracts the raw and cleaned data from the dataframe argument and stores them in the corresponding directory offered as an argument."""
    raw_transcript_records_df = transcript_records[["transcript_id", "filename", "platform", "raw_text"]].copy()
    cleaned_transcript_records_df = transcript_records[["transcript_id", "platform", "cleaned_text"]].copy()
    
    filepath = raw_data_dir / "raw_transcript_records.csv"
    if filepath.exists():
       styling_and_animations.print_warning_message(f"Raw CSV already exists, thus will skip this step ({filepath})")
    else:
        store_as_csv(raw_transcript_records_df, filepath)
        
    filepath = cleaned_data_dir / "cleaned_transcript_records.csv"
    if filepath.exists():
       styling_and_animations.print_warning_message(f"Cleaned CSV already exists, thus will skip this step ({filepath})")
    else:
        store_as_csv(cleaned_transcript_records_df, filepath)

    return None


def store_as_csv(df: pd.DataFrame, full_filepath: Path) -> None:
    """Stores the dataframe argument as a CSV file in the given directory. """
    try:
        if full_filepath.exists():
           styling_and_animations.print_warning_message(f"This CSV file already exists, thus will skip this step ({full_filepath})")
        else:
            df.to_csv(str(full_filepath), index=False, encoding="utf-8")
    except Exception as e:
        styling_and_animations.print_error_message(f"Failed (Error: {type(e).__name__}) to export data ({full_filepath}):", e)
         
    return None
    
    