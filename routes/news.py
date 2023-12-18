# Imports needed in news.py
from flask import request
from services.sentiment_analysis import analyze_sentiment
from datetime import datetime, timedelta
import requests
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def fetchnews():
    url = 'https://api.phoenixnews.io/getLastNews?limit=200'
    response = requests.get(url)
    data = response.json()
    df = filtertje(pd.DataFrame(data))

    # List of columns to remove
    df['Text'] = df['body'].fillna('') + ' ' + df['description'].fillna('')
    columns_to_remove = ['_id', 'body', 'image3', 'image4', 'imageQuote2', 'imageQuote3', 'imageQuote4', 'image',
                         'description', 'createdAt', 'url', 'title', 'suggestions', 'category', 'isReply', 'coin',
                         'image1', 'username', 'name', 'icon', 'twitterId', 'tweetId', 'isRetweet', 'isQuote',
                         'image', 'imageQuote', 'image2', "important"]

    # Drop specified columns
    df = df.drop(columns=columns_to_remove, errors='ignore')
    df['score'] = df['Text'].apply(analyze_sentiment)
    df['receivedAt'] = df['receivedAt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d %H:%M:%S'))
    df['receivedAt'] = pd.to_datetime(df['receivedAt'])

    global_df['receivedAt'] = df['receivedAt']
    global_df['receivedAt'] = global_df['receivedAt'] + timedelta(hours=1)
    global_df['score'] = df['score']

    # Fetching the table HTML
    table_html = df.to_html(classes='table', index=False)

    # Add classes to score cells in the HTML table
    for index, row in df.iterrows():
        score_class = 'neutral'
        if row['score'] > 0.5:
            score_class = 'positive'
        elif row['score'] < 0.5:
            score_class = 'negative'

        table_html = table_html.replace(f'<td>{row["score"]}</td>', f'<td class="color-{score_class}">{row["score"]}</td>')

    # Extract only the tbody content
    tbody_start = table_html.find('<tbody>')
    tbody_end = table_html.find('</tbody>') + len('</tbody>')
    tbody_content = table_html[tbody_start:tbody_end]

    return tbody_content

global_df = pd.DataFrame(columns=['receivedAt', 'score'])

def filtertje(df):
    # Drop rows where 'Source' is equal to 'Twitter'
    # Assuming 'source' column is in lowercase, adjust accordingly if needed
    df_filtered = df[(df['source'] != 'Twitter')  & (df['body'] != '')].copy()

    return df_filtered
