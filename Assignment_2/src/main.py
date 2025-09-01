#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 17:05:06 2025

@author: constantina

"""

# https://developers.google.com/youtube/v3/docs/?apix=true

import pandas as pd
from custom_packages.misc_modules import config, visualisations
from custom_packages.preprocessing_modules import data_collector, transcriber, text_cleaning
from custom_packages.modelling_modules import nlp_modelling

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping


# I change these values depending on whether I've already pulled all the videos I want from Youtube and stored them as JSON files or not yet;
# and if I have completed transcribing all videos or not yet.
json_files_exist = True 
transcriptions_completed = True
cleaned_data_csv_exists = True
transcripts_filepath_dict = {}

# if transcription is completed then the only thing remaining to do is to
# collect the raw transcripts to clean them and prepare them for sentiment/trust
# analysis and topic modelling and plotting.
if transcriptions_completed:
    for platform_name in config.DATA_PLATFORMS_EXAMINED:
        transcripts_filepath_dict[platform_name] = data_collector.get_transcripts(config.transcript_output_dir, platform_name)

# otherwise, if the transcripts don't exist yet, but the Youtube json files exist,
# then retrieve the json files and start the transcription process.
elif json_files_exist:
    video_df = data_collector.get_json_data(config.json_dir)
    if video_df is None:
        print("One or more json files could not be retrieved or processed, check and restart.")
    else:
        print(video_df.head())
        print("\nTotal videos retrieved:" + str(len(video_df))) # 198 videos found in total.
        
        # Starting the transcription process.
        transcriber.transcribe_all_videos(video_df)

# otherwise, if json files don't exist yet either, start from scratch ...
else: # Get youtube data (based on search query keywords) to be stored as json files, and initiate the transcription process.
    # These search terms were taken from "trustworthy" and "untrustworthy" term synonyms suggested on
    # the Merriam-Webster Thesaurus website (https://www.merriam-webster.com/thesaurus/trustworthy)
    # in August 2025.
    # I had used WordNet and the NRC emotion lexicon in a previous attempt, but
    # Merriam-Webster Thesaurus seems to offer more relevant synonyms in my case.
    synonym_groups = {
        "trust": ("trustworthy AI", "reliable AI", "responsible AI", "dependable AI", "safe AI"),
        "distrust": ("distrustful AI", "fraudulent AI", "dishonest AI", "deceptive AI", "lying AI")
    }
    video_df = data_collector.store_all_video_data(synonym_groups, max_results=100)
    print(video_df.head())
    print("\nTotal videos retrieved:" + str(len(video_df)))
    
    # Starting the transcription process.
    transcriber.transcribe_all_videos(video_df)
    
    for platform_name in config.DATA_PLATFORMS_EXAMINED:
        transcripts_filepath_dict[platform_name] = data_collector.get_transcripts(config.transcript_output_dir, platform_name)
        

# This stage is reached only when transcriptions_completed == True AND cleaned_data_csv_exists == True 

# Now, if only transcriptions_completed but not cleaned_data_csv_exists, then get on with cleaning the data
# before getting on with modelling.
if not cleaned_data_csv_exists:
    cleaned_transcript_records_df = text_cleaning.clean_transcripts(transcripts_filepath_dict)
    text_cleaning.store_cleaned_data_as_csv(cleaned_transcript_records_df, config.my_base_dir_raw, config.my_base_dir_cleaned)


# This stage is reached only when transcriptions_completed == True AND cleaned_data_csv_exists == True.
# Only thing remaining to do now is run the models for DFM, sentiment/trust, and topic modelling - including visualisations.    
cleaned_data_df = data_collector.get_cleaned_csv_record(config.my_base_dir_cleaned / "cleaned_transcript_records.csv")
dfm = nlp_modelling.get_document_feature_matrix(cleaned_data_df)
visualisations.plot_word_frequencies(dfm, words_num = 30)
visualisations.word_cloud_visualisation(dfm, words_num = 100)

# ideally run these in order before proceeding with topic modelling - otherwise, it will break.
cleaned_data_with_sentiment_df = nlp_modelling.sentiment_analysis_using_nltk(cleaned_data_df)
cleaned_data_with_sentiment_df = nlp_modelling.sentiment_analysis_using_distilbert(cleaned_data_with_sentiment_df)
cleaned_data_with_sentiment_df = nlp_modelling.sentiment_analysis_using_gpt(cleaned_data_with_sentiment_df)
cleaned_data_with_sentiment_df = nlp_modelling.sentiment_analysis_using_google(cleaned_data_with_sentiment_df)

cleaned_data_with_all_model_results_df = nlp_modelling.get_dominant_topic(cleaned_data_with_sentiment_df)

text_cleaning.store_as_csv(cleaned_data_with_all_model_results_df, config.my_base_dir_cleaned / "cleaned_data_with_all_model_results.csv")

nlp_modelling.sentiment_model_comparison(cleaned_data_with_all_model_results_df)
    




    





