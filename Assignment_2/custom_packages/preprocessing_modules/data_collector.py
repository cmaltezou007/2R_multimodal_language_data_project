#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 18:31:53 2025

@author: constantina
"""

import pandas as pd
from pathlib import Path
import json
import time
from custom_packages.misc_modules import styling_and_animations, config


def store_all_video_data(search_query_groups: dict, max_results: int = 10) -> pd.DataFrame:
    """
    Uses the given dictionary of search query strings to retrieve relevant Youtube videos,
    and store them locally as json files.

    Parameters
    ----------
    search_query_groups : dict
        A dictionary of strings, for all search queries per search group (e.g., trust/distrust groups).
    max_results : int, optional
        The maximum number of total video results to retrieve and store. 
        The default is 10.

    Raises
    ------
    ValueError
        Raised if no Youtube data retrieved / empty dataframe.

    Returns
    -------
    pd.DataFrame
        Contains all the extracted video data.

    """
    video_df = pd.DataFrame()
    
    # Going through all search terms to retrieve all relevant video data results, while removing any duplicates.
    for query_group_name, synonyms in search_query_groups.items():
        search_terms = "|".join(synonyms)  # OR encoded: search_terms = "%7C".join(trust_set)
        print(f"\nSearch terms for query {query_group_name}:", search_terms)
        temp_df = get_youtube_video_data(search_terms, query_group_name, max_results)
        print(f"\ntemp_df length for query {query_group_name}:" + str(len(temp_df)))
        print(f"Unique video IDs in temp_df ({query_group_name}):", temp_df['video_id'].nunique())
        
        video_df = pd.concat([video_df, temp_df])
        print("\nTotal unique video IDs in video_df so far:", video_df['video_id'].nunique())
        print(f"video_df length for query {query_group_name} (before duplicates removal):" + str(len(video_df)))
        video_df = video_df.drop_duplicates(subset="video_id").reset_index(drop=True)
        print(f"video_df length for query {query_group_name} (after duplicates removal):" + str(len(video_df)))
        
        if video_df.empty:
            raise ValueError("The dataframe for retrieved video data is empty!")
        
    return video_df


def get_youtube_video_data(query: str, query_group_name: str, max_results: int = 10) -> pd.DataFrame:
    """
    1. Uses Youtube API v3 to retrieve the list of relevant videos based on the function parameters
    2. Extract a list of video data to return.
    3. Store all of the retrieved search data locally as a json file on the computer.

    Parameters
    ----------
    query : str
        The search term to lookup on Youtube.
    query_group_name : str
        A unique identifier for this query - used in the exported json filename.
    max_results : int (optional; default value of 10)
        Specifies specifies the maximum number of items(video) that should be returned.

    Returns
    -------
    pd.Dataframe
        Contains all the extracted video data.
    """
    
    videos = []
    all_items = []
    next_page_token = None

    # Running a while loop to make sure I get the precise number of results I want, 
    # as a good practice to be wary of any repercussions on my quota or costs.
    while len(all_items) < max_results:
    
        try:  # get video data between 5 - 20 mins long.
            search_response = config.get_youtube_client().search().list(
                q = query, # it can accept: q = "cats|dogs|parrots"
                type = "video",
                part = "id,snippet",
                videoDuration = "medium", # or "any"
                maxResults = max_results - len(all_items), #Youtube API's default is 5.
                pageToken = next_page_token
            ).execute() 
        
            json_output_filepath = config.json_dir / f"youtube_video_search_results_{query_group_name}_{next_page_token}.json"
            with open(json_output_filepath, "w", encoding="utf-8") as jsonfile:
                json.dump(search_response, jsonfile, indent = 2)
        except TypeError as e:
            styling_and_animations.print_error_message(f"json serialisation error ({type(e).__name__}):", e)
        except Exception as e:
            styling_and_animations.print_error_message(f"Error while saving ({type(e).__name__}):", e)
            #traceback.print_exc()
        
        
        for item in search_response['items']:
            video_id = item['id']['videoId']
            video_title = item['snippet']['title']
            description = item['snippet']['description']
            channel = item['snippet']['channelTitle']
            publish_time = item['snippet']['publishedAt']
            videos.append({
                'video_id': video_id,
                'title': video_title,
                'description': description,
                'channel': channel,
                'publish_time': publish_time,
                'url': f"https://www.youtube.com/watch?v={video_id}"
            })
        
        time.sleep(0.3)
        all_items.extend(search_response["items"])

        next_page_token = search_response.get("nextPageToken")
        if not next_page_token:
            return pd.DataFrame(videos)
        
    return pd.DataFrame(videos)



def get_json_data(files_dir: Path) -> pd.DataFrame:
    """
    Get all json files from given directory and return a subset of their data 
    in a dataframe format.
    
    Parameters
    ----------
    files_dir : Path.PosixPath
        The directory where all the json files are stored.

    Returns
    -------
    pd.DataFrame
        Contains json data such as video_id, url, publish_time, etc.

    """
    json_files = [filepath for filepath in files_dir.iterdir() if filepath.suffix == ".json"]
    video_data = []
    
    for file_path in json_files:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    
        for item in data.get("items", []):
            try:
                video_id = item['id']['videoId']
                video_title = item['snippet']['title']
                description = item['snippet']['description']
                channel = item['snippet']['channelTitle']
                publish_time = item['snippet']['publishedAt']
                video_data.append({
                    'video_id': video_id,
                    'title': video_title,
                    'description': description,
                    'channel': channel,
                    'publish_time': publish_time,
                    'url': f"https://www.youtube.com/watch?v={video_id}"
                })
    
            except KeyError as e:
                styling_and_animations.print_error_message(f"Skipped {item} entry in {file_path} due to missing key (Error: {type(e).__name__}):", e)
                return None
                
    video_df = pd.DataFrame(video_data).drop_duplicates(subset="video_id").reset_index(drop=True)
    print(f"Loaded {len(video_df)} unique videos from {len(json_files)} files.")
    return video_df


def get_transcripts(files_dir: Path, platform_name: str) -> list:
    """
    Get all transcript .txt files from given directory and combine their text 
    in a single string output.
    
    Parameters
    ----------
    files_dir : Path.PosixPath
        The directory where all the transcription .txt files are stored.

    Returns
    -------
    str
        Contains the combined text from all the .txt files.

    """
    
    styling_and_animations.print_animated_ellipsis_message("Retrieving all stored transcripts")
    text_files = [filepath for filepath in files_dir.iterdir() if filepath.suffix == ".txt"]
    
    return text_files


def get_cleaned_csv_record(csv_filepath: Path) -> pd.DataFrame:
    """Retrieves a CSV file as a dataframe from the given path. """
    return pd.read_csv(csv_filepath)
    
    
    