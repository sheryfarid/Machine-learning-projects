from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from emoji import is_emoji
import os
import mimetypes
from PIL import Image
from mutagen import File as AudioFile
import re
import string

extract = URLExtract()

# Load Hinglish stopwords
stopfile = os.path.join(os.path.dirname(__file__), 'stop_hinglish.txt')
with open(stopfile, 'r') as f:
    stop_words = f.read().splitlines()

block_words = set([
    'omitted', 'attached', 'image', 'file', 'video', 'audio', 'document',
    'sticker', 'webp', 'jpg', 'jpeg', 'png', 'mp4', 'opus', 'm4a', 'mp3', '3gp',
    'img', 'vid', 'ptt', 'aud', 'stk', 'doc', 'pdf', 'xlsx', 'pptx', 'docx'
])

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = [word for message in df['message'] for word in message.split()]
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = [url for message in df['message'] for url in extract.find_urls(message)]
    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df

def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.contains('<Media omitted>', na=False)]

    def clean_message(message):
        words = []
        for word in message.lower().split():
            word = word.strip(string.punctuation)
            if any(sub in word for sub in block_words):
                continue
            if word and word not in stop_words:
                words.append(word)
        return " ".join(words)

    temp['message'] = temp['message'].apply(clean_message)
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    return wc.generate(temp['message'].str.cat(sep=" "))

def most_common_words(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.contains('<Media omitted>', na=False)]

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            word = word.strip(string.punctuation)
            if word and word not in stop_words and not any(sub in word for sub in block_words):
                words.append(word)
    return pd.DataFrame(Counter(words).most_common(20))

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = [c for message in df['message'] for c in message if is_emoji(c)]
    return pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.groupby('only_date').count()['message'].reset_index()

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

def get_user_media_stats(df):
    media_keywords = ['IMG-', 'VID-', 'PTT-', 'STK-', 'AUD-', 'DOC-']
    user_media_counts = {}

    for _, row in df.iterrows():
        for keyword in media_keywords:
            if keyword in row['message']:
                user = row['user']
                user_media_counts.setdefault(user, {})
                user_media_counts[user][keyword] = user_media_counts[user].get(keyword, 0) + 1
    return pd.DataFrame(user_media_counts).fillna(0).astype(int).T

def get_user_media_breakdown(df):
    media_data = []

    for _, row in df.iterrows():
        message = row['message']
        user = row['user']
        matches = re.findall(
            r'\b(?:IMG|VID|PTT|STK|AUD|DOC)-[^\s]+\.(?:jpg|jpeg|png|webp|mp4|3gp|mpeg4|opus|mp3|m4a|pdf|docx|xlsx|pptx)\b',
            message, flags=re.IGNORECASE
        )
        for filename in matches:
            prefix = filename.upper().split("-")[0]
            ext = os.path.splitext(filename)[1].lower()
            media_type = {
                "STK": "Stickers",
                "IMG": "Images",
                "VID": "Videos",
                "PTT": "Audio",
                "AUD": "Audio",
                "DOC": "Documents"
            }.get(prefix, None)
            if media_type:
                media_data.append((user, media_type, row['only_date'], row['month'], row['day_name'], row['period']))

    if not media_data:
        return pd.DataFrame()
    return pd.DataFrame(media_data, columns=['user', 'type', 'date', 'month', 'day', 'period'])

def count_media_placeholders(df):
    return df['message'].str.contains('<Media omitted>', na=False).sum()

def get_unique_media_summary_from_chat(df):
    patterns = {
        "Stickers": r'STK-\d{8}-WA\d+\.(webp)',
        "Images": r'IMG-\d{8}-WA\d+\.(jpg|jpeg|png|webp)',
        "Videos": r'VID-\d{8}-WA\d+\.(mp4|3gp|mpeg4)',
        "Audio": r'PTT-\d{8}-WA\d+\.(opus|m4a|mp3)',
        "Documents": r'(DOC-\d{8}-WA\d+\.(pdf|docx|xlsx|txt|zip|rar|pptx|csv|html|json))|(\b[\w\-]+\.pdf\b)|(\b[\w\-]+\.docx\b)|(\b[\w\-]+\.xlsx\b)'
,
        "GIFs": r'GIF-\d{8}-WA\d+\.(mp4|gif)'
    }
    counts = {}
    for media_type, pattern in patterns.items():
        matches = df['message'].str.extractall(f"({pattern})")[0].dropna().unique()
        counts[media_type] = len(matches)
    return pd.DataFrame(list(counts.items()), columns=["Media Type", "Unique Count"])

def get_user_media_type_counts_from_chat(df):
    media_keywords = {
        'Images': 'IMG-',
        'Videos': 'VID-',
        'Audio': 'PTT-',
        'Stickers': 'STK-'
    }
    media_data = []
    for _, row in df.iterrows():
        for media_type, keyword in media_keywords.items():
            if keyword in row['message']:
                media_data.append({'User': row['user'], 'Media Type': media_type})
    if not media_data:
        return pd.DataFrame()
    media_df = pd.DataFrame(media_data)
    return media_df.pivot_table(index='User', columns='Media Type', aggfunc='size', fill_value=0).reset_index()
