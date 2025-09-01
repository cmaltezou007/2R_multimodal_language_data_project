#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 20:43:57 2025

@author: constantina
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.decomposition import PCA
import numpy as np


def plot_word_frequencies(df: pd.DataFrame, words_num: int) -> None:
    """
    Shows a bar plot with the most frequent words across all transcripts.

    Parameters
    ----------
    df : pd.DataFrame
        A document-ffeature matrix (DFM) with keyword frequencies.
    words_num : int
        How many highest-frequency keywords to show in the plot.
        
    Returns
    -------
    None

    """
    word_frequencies = df.sum(axis=0).sort_values(ascending=False).head(words_num)
    
    plt.figure(figsize=(12, 6))
    word_frequencies.plot(kind="bar", rot=50)
    plt.title(f"Top {words_num} most frequent words, across all transcripts")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    return None


def word_cloud_visualisation(df: pd.DataFrame, words_num: int) -> None:
    """
    Visualises a word cloud of the top keywords across all transcripts.

    Parameters
    ----------
    df : pd.DataFrame
        A document-ffeature matrix (DFM) with keyword frequencies.
    words_num : int
        How many highest-frequency keywords to show in the word cloud.

    Returns
    -------
    None

    """
    
    word_frequencies = df.sum(axis=0).sort_values(ascending=False).head(words_num)
    
    word_cloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Top word frequencies across all transcripts")
    plt.show()
    return None


def plot_sentiment_distribution(df: pd.DataFrame, sentiment_column: str) -> None:
    "Shows a bar plot of the count distribution of sentiment labels across all transcripts."
    plt.figure(figsize=(6,4))
    sns.countplot(x=sentiment_column, data=df, palette="Set1")
    plt.title(f"{sentiment_column} distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of transcripts")
    plt.tight_layout()
    plt.show()
    return None


def plot_trust_distribution(df: pd.DataFrame, sentiment_column: str) -> None:
    "Shows a bar plot of the count distribution of trust labels across all transcripts."
    plt.figure(figsize=(6,4))
    sns.countplot(x=sentiment_column, data=df, palette="Set2")
    plt.title(f"Mapped {sentiment_column} distribution")
    plt.xlabel("Trust label")
    plt.ylabel("Number of transcripts")
    plt.tight_layout()
    plt.show()
    return None


def plot_topic_clusters(transcript_topic_distribution: np.ndarray, dominant_topic_labels: np.ndarray) -> None:
    """Shows a scatter plot that groups transcripts with similar topics. """
    pca = PCA(n_components=2)
    coords = pca.fit_transform(transcript_topic_distribution)
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=coords[:,0], y=coords[:,1], hue=dominant_topic_labels, palette="tab20", legend=False)
    plt.title("Topic clusters (PCA)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(title="Dominant topic")
    plt.tight_layout()
    plt.show()
    return None


def plot_sentiment_by_topic(df: pd.DataFrame, trust_column_name: str, topic_column_name: str) -> None:
    """Shows a bar plot that shows the distribution of trust scores from a model for each topic category. """
    topic_trust_counts = df.groupby([topic_column_name, trust_column_name]).size().unstack().fillna(0)
    topic_trust_counts = topic_trust_counts.div(topic_trust_counts.sum(axis=1), axis=0)  # normalize to %
    
    topic_trust_counts.plot(kind="bar", stacked=True, figsize=(10,6), colormap="coolwarm", rot=0)
    plt.title(f"{trust_column_name} distribution by topic")
    plt.xlabel("Topic ID")
    plt.ylabel("Proportion")
    plt.legend(title="Trust label")
    plt.tight_layout()
    plt.show()
    return None
