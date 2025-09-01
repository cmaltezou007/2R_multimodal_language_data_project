#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 20:05:12 2025

@author: constantina
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
from transformers import pipeline
from fastopic import FASTopic
from topmost.preprocess import Preprocess
import numpy as np
from custom_packages.misc_modules import visualisations, config, styling_and_animations
import json
import time
from datetime import datetime
from google.genai import types

#nltk.download('vader_lexicon')


def get_document_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a basic bag-of-words document-feature matrix (DFM) from the given cleaned text in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Needs to contain a column named "cleaned_text".

    Returns
    -------
    dfm : pd.DataFrame

    """
    vectoriser = CountVectorizer()
    X = vectoriser.fit_transform(df["cleaned_text"])
    
    dfm = pd.DataFrame(X.toarray(), columns=vectoriser.get_feature_names_out())
    
    return dfm


def sentiment_analysis_using_nltk(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    1. Retrieves the partly cleaned/preprocessed text (decoupling contranctions, converting to lowercase and removing
        dashes and unnecessary white spaces already done earlier) from the dataframe argument.
    2. Finishes preprocessing by removing punctuation, stopwords, and applies tokenisation through NLTK.
    3. Does sentiment analysis using NLTK Vader SentimentIntensityAnalyzer on the tokenised text, and calculates the 
        polarity scores (positive, negative, neutral).
    4. Visualises the sentiment and translated trust labels in a plot each.
    5. Returns a dataframe with the scores and labels.
    """
    
    vader = SentimentIntensityAnalyzer()
    vader_sentiments = []
    vader_scores = []
    vader_trust = []
    
    for i, row in df.iterrows():
        styling_and_animations.print_animated_ellipsis_message(f"Processing item {i+1}({len(df)} total) for sentiment analysis with NLTK Vader")
        text = row["cleaned_text"]
        
        # Additional cleaning to remove punctuation and stopwords, and apply tokenisation.
        text = re.sub(r"[^\w\s]", "", text)
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        tokenised_text = " ".join(tokens)
        
        try:
            scores_dict = vader.polarity_scores(tokenised_text)
            compound_score = scores_dict["compound"]
            sentiment = (
                "positive" if compound_score >= 0.05 else
                "negative" if compound_score <= -0.05 else
                "neutral"
            )
            trust_label = map_sentiment_to_trust(sentiment)
            
            vader_sentiments.append(sentiment)
            vader_scores.append(compound_score)
            vader_trust.append(trust_label)
            
        except Exception as e:
            styling_and_animations.print_error_message(f"Error ({type(e).__name__}); something went wrong with the response from NLTK Vader:", e)
            break
        
    df["vader_sentiment"] = vader_sentiments
    df["vader_score"] = vader_scores
    df["vader_trust"] = vader_trust
    
    visualisations.plot_sentiment_distribution(df, "vader_sentiment")
    visualisations.plot_trust_distribution(df, "vader_trust")

    return df


def sentiment_analysis_using_distilbert(df: pd.DataFrame) -> None:
    """ 
    1. Retrieves the partly cleaned/preprocessed text (decoupling contranctions, converting to lowercase and removing
        dashes and unnecessary white spaces already done earlier) from the dataframe argument. No need for any further
        preprocessing nor tokenisation as DistilBert handles that internally.
    2. Does sentiment analysis using the DistilBert sentiment pipeline and calculates the polarity scores 
        (positive, negative).
    3. Visualises the sentiment and translated trust labels in a plot each.
    4. Returns a dataframe with the scores and labels.
    """
    
    transformer_senti = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    transformer_sentiments = []
    transformer_scores = []
    transformer_trust = []
    
    for i, row in df.iterrows():
        styling_and_animations.print_animated_ellipsis_message(f"Processing item {i+1}({len(df)} total) for sentiment analysis with distilbert-base-uncased-finetuned-sst-2-english")
        text = row["cleaned_text"]
        
        try:
            # It seems that distilbert can only accept up to 512 tokens at a time,
            # otherwise it complains.
            transformer_output = transformer_senti(text[:500], batch_size = 10)[0]  # limit for speed/memory
            sentiment_label = transformer_output["label"].lower()  # "POSITIVE", "NEGATIVE"
            score = transformer_output["score"]
            trust_label = map_sentiment_to_trust(sentiment_label)
            
            transformer_sentiments.append(sentiment_label)
            transformer_scores.append(score)
            transformer_trust.append(trust_label)
            
            time.sleep(0.3)
            
        except Exception as e:
            styling_and_animations.print_error_message(f"Error ({type(e).__name__}); something went wrong with the response from DistilBERT:", e)
            break
        
    df["distilbert_sentiment"] = transformer_sentiments
    df["distilbert_score"] = transformer_scores
    df["distilbert_trust"] = transformer_trust
        
    visualisations.plot_sentiment_distribution(df, "distilbert_sentiment")
    visualisations.plot_trust_distribution(df, "distilbert_trust")

    return df


def sentiment_analysis_using_gpt(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    1. Retrieves the partly cleaned/preprocessed text (decoupling contranctions, converting to lowercase and removing
        dashes and unnecessary white spaces already done earlier) from the dataframe argument. No need for any further
        preprocessing nor tokenisation as gpt-4o-mini handles that internally.
    2. Does sentiment analysis using the gpt-4o-mini model (requires OpenAI API) and responds with json-formatted 
        results on polarity scores (positive, negative, neutral).
    3. Visualises the sentiment and translated trust labels in a plot each.
    4. Returns a dataframe with the scores, labels, and summary highlights.
    """
    
    genai_sentiments = []
    genai_highlights = []
    genai_trust = []
    genai_analyses = []
    genai_datetime = []
    
    for i, row in df.iterrows():
        styling_and_animations.print_animated_ellipsis_message(f"Processing item {i+1}({len(df)} total) for sentiment analysis with gpt-4o-mini")
        text = row["cleaned_text"]
        
        prompt = f"""Please analyse and report the overall sentiment of the following transcript in terms of public discourse relating to the trustworthiness of artificial intelligence (AI). 
            When you report the overall sentiment you should strictly use only one sentiment label out of these three: [POSITIVE, NEGATIVE, NEUTRAL].
        
            Transcript to analyse:
                "{text}"
            
            You should return the output strictly in the following JSON format:
                {{
                    "Sentiment_label": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
                    "Highlights": ["...", "...", "..."],
                    "Sentiment_analysis": "..."
                }}
        """
        
        try:
            # Useful documentation links: 
            # https://platform.openai.com/docs/guides/text
            # https://platform.openai.com/docs/guides/migrate-to-responses
            response = config.get_openai_client().responses.create(
                model = "gpt-4o-mini",
                input = [{"role": "user", "content": prompt}],
                temperature = 0 # I'm trying to reduce its non-deterministic nature, for partial reporducibility.
            )
            
            cleaned_output_for_json = re.sub(r"^```json|```$", "", response.output_text.strip(), flags=re.MULTILINE).strip()
            parsed_response = json.loads(cleaned_output_for_json)
            
            trust_label = map_sentiment_to_trust(parsed_response.get("sentiment_label", "NA").lower())
            
            genai_sentiments.append(parsed_response.get("sentiment_label", "NA").lower())
            genai_highlights.append(parsed_response.get("highlights", []))
            genai_analyses.append(parsed_response.get("sentiment_analysis", ""))
            genai_trust.append(trust_label)
            genai_datetime.append(str(datetime.utcnow()))
            
        except Exception as e:
            styling_and_animations.print_error_message(f"Error ({type(e).__name__}); something went wrong with the response from GPT:", e)
            break
    
    df["gpt4omini_sentiment"] = genai_sentiments
    df["gpt4omini_trust"] = genai_trust
    df["gpt4omini_highlights"] = genai_highlights
    df["gpt4omini_analyses"] = genai_analyses
    
    visualisations.plot_sentiment_distribution(df, "gpt4omini_sentiment")
    visualisations.plot_trust_distribution(df, "gpt4omini_trust")
    
    return df


def sentiment_analysis_using_google(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    1. Retrieves the partly cleaned/preprocessed text (decoupling contranctions, converting to lowercase and removing
        dashes and unnecessary white spaces already done earlier) from the dataframe argument. No need for any further
        preprocessing nor tokenisation as gemini-2.0-flash-lite handles that internally.
    2. Does sentiment analysis using the gemini-2.0-flash-lite model (requires Google API) and responds with 
        json-formatted results on polarity scores (positive, negative, neutral).
    3. Visualises the sentiment and translated trust labels in a plot each.
    4. Returns a dataframe with the scores, labels, and summary highlights.
    """
    
    genai_sentiments = []
    genai_highlights = []
    genai_trust = []
    genai_analyses = []
    genai_datetime = []
    
    for i, row in df.iterrows():
        styling_and_animations.print_animated_ellipsis_message(f"Processing item {i+1}({len(df)} total) for sentiment analysis with gemini-2.0-flash")
        text = row["cleaned_text"]
        
        prompt = f"""Please analyse and report the overall sentiment of the following transcript in terms of public discourse relating to the trustworthiness of artificial intelligence (AI). 
            When you report the overall sentiment you should strictly use only one sentiment label out of these three: [POSITIVE, NEGATIVE, NEUTRAL].
        
            Transcript to analyse:
                "{text}"
            
            You should return the output strictly in the following JSON format:
                {{
                    "sentiment_label": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
                    "highlights": ["...", "...", "..."],
                    "sentiment_analysis": "..."
                }}
        """
        
        try:
            response = config.get_gemini_client().models.generate_content(
                model="gemini-2.0-flash-lite",
                contents = prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1
                )
            )
            
            cleaned_output_for_json = re.sub(r"^```json|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
            parsed_response = json.loads(cleaned_output_for_json)
            
            trust_label = map_sentiment_to_trust(parsed_response.get("sentiment_label", "NA").lower())
            
            genai_sentiments.append(parsed_response.get("sentiment_label", "NA").lower())
            genai_highlights.append(parsed_response.get("highlights", []))
            genai_analyses.append(parsed_response.get("sentiment_analysis", ""))
            genai_trust.append(trust_label)
            genai_datetime.append(str(datetime.utcnow()))
            
            time.sleep(0.3)
            
        except Exception as e:
            styling_and_animations.print_error_message(f"Error ({type(e).__name__}); something went wrong with the response from Gemini:", e)
            break
    
    df["gemini_sentiment"] = genai_sentiments
    df["gemini_trust"] = genai_trust
    df["gemini_highlights"] = genai_highlights
    df["gemini_analyses"] = genai_analyses
    
    visualisations.plot_sentiment_distribution(df, "gemini_sentiment")
    visualisations.plot_trust_distribution(df, "gemini_trust")
    
    return df
    

def sentiment_model_comparison(df: pd.DataFrame) -> None:
    """ 
    The dataframe argument should contain the trust sentiment values from all models.
    Compares and displays in the console the agreement rate between all models.
    """
    print(df["vader_trust"].value_counts())
    print("\n", df["distilbert_trust"].value_counts())
    print("\n", df["gpt4omini_trust"].value_counts())
    print("\n", df["gemini_trust"].value_counts())
    
    print_models_agreement_percentage(df["vader_trust"], df["distilbert_trust"])
    print_models_agreement_percentage(df["vader_trust"], df["gpt4omini_trust"])
    print_models_agreement_percentage(df["vader_trust"], df["gemini_trust"])
    
    print_models_agreement_percentage(df["distilbert_trust"], df["gpt4omini_trust"])
    print_models_agreement_percentage(df["distilbert_trust"], df["gemini_trust"])
    
    print_models_agreement_percentage(df["gpt4omini_trust"], df["gemini_trust"])
    
    return None

def print_models_agreement_percentage(first_model: pd.Series, second_model: pd.Series) -> None:
    """Calculates and displays in the console the agreement rate between two models. """
    agreement_percentage = (first_model == second_model).mean()
    agreement_percentage = round(agreement_percentage * 100, 2)
    print(f"\nAgreement rate between {first_model.name} and {second_model.name}: {agreement_percentage}%")
    
    return None


def map_sentiment_to_trust(sentiment_label: str) -> str:
    """Converts classic sentiment labels to trust/distrust/neutral. """
    if sentiment_label in ["label_1", "positive", "favor"]:
        return "trust"
    elif sentiment_label in ["label_0", "negative", "against"]:
        return "distrust"
    elif sentiment_label in ["na", "n/a"]:
        return "NA"
    
    return "neutral"


def get_dominant_topic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Does topic modelling using FASTopic on the cleaned text from each transcript and retrieves teh top keywords
    identified per topic (5 most popular topics in this instance).
    Displays a plot per sentiment model for a trust-sentiment score by topic assessment.
    It also displays another plot that clusters transcripts with similar topics.
    """
    transcripts = df["cleaned_text"].tolist()
    preprocess = Preprocess() # Handles tokenisation, stopwords, etc.
    model = FASTopic(5, preprocess)
    
    # extracts top words per topic, and the probability distribution of topics 
    # for each transcript.
    top_words, transcript_topic_distribution = model.fit_transform(transcripts)
    
    for i, topic in enumerate(top_words):
        print(f"\nTopic {i+1}: {', '.join(topic.split())}")
    
    dominant_topic_index_per_transcript = np.argmax(transcript_topic_distribution, axis = 1)
    df["dominant_topic_index"] = dominant_topic_index_per_transcript+1
    
    visualisations.plot_topic_clusters(transcript_topic_distribution, dominant_topic_index_per_transcript)
    visualisations.plot_sentiment_by_topic(df, "vader_trust", "dominant_topic_index")
    visualisations.plot_sentiment_by_topic(df, "distilbert_trust", "dominant_topic_index")
    visualisations.plot_sentiment_by_topic(df, "gpt4omini_trust", "dominant_topic_index")
    visualisations.plot_sentiment_by_topic(df, "gemini_trust", "dominant_topic_index")
    
    return df

    



    
    
    
    

