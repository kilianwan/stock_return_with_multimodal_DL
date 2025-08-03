import pandas as pd
import numpy as np
import torch

def get_sentiment_score(text, sentiment_pipeline, max_chunk_words=250):
    words = text.split()
    chunks = [' '.join(words[i: i+ max_chunk_words]) for i in range(0, len(words), max_chunk_words)]
    results = sentiment_pipeline(chunks)
    
    def score_from_label(result):
        label = result['label'].lower()
        score = result['score']
        if 'positive' in label:
            return score
        elif 'negative' in label:
            return -score
        else:
            return 0.0
    
    signed_scores = [score_from_label(r) for r in results]
    
    return np.mean(signed_scores) if signed_scores else 0.0

def map_sentiment(avg_sentiment, threshold):
    if abs(avg_sentiment) < threshold:
        return 0
    else: return np.sign(avg_sentiment)

def get_signals(df, nlp, threshold):
    sentiment_scores = []

    for _, row in df.iterrows():
        text = row["text"]
        score = get_sentiment_score(text, nlp)
        
        if threshold is not None:
            score = map_sentiment(score, threshold)
            
        sentiment_scores.append(score)

    df = df.copy()
    df["sentiment_score"] = sentiment_scores
    df.drop(columns="text", inplace=True)
    df.set_index(['cusip', 'year', 'quarter', 'gvkey'], inplace=True)
    return df