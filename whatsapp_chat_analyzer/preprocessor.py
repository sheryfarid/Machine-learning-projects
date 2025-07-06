import pandas as pd
import re
from datetime import datetime


def remove_media_placeholders(text):
    placeholders = [
        'file attached', 'image omitted', 'video omitted', 'audio omitted',
        'attached', 'sticker omitted', 'document omitted'
    ]
    text = text.lower()
    for ph in placeholders:
        text = text.replace(ph, '')
    return text


def hour_to_readable_period(hour):
    start = int(hour)
    end = (start + 1) % 24
    def label(h):
        if h == 0: return "12am"
        elif h < 12: return f"{h}am"
        elif h == 12: return "12pm"
        else: return f"{h-12}pm"
    return f"{label(start)} - {label(end)}"


def preprocess(data):
    pattern = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2})[\u200e\u202f ]?(AM|PM) - '

    messages = re.split(pattern, data)[1:]
    dates, users, texts = [], [], []

    for i in range(0, len(messages), 4):
        try:
            date_str = messages[i]
            time_str = messages[i+1]
            ampm = messages[i+2]
            msg = messages[i+3]

            datetime_obj = datetime.strptime(f"{date_str}, {time_str} {ampm}", "%m/%d/%y, %I:%M %p")
            dates.append(datetime_obj)

            if ": " in msg:
                user, message = msg.split(": ", 1)
            else:
                user = "group_notification"
                message = msg

            users.append(user.strip())
            texts.append(message.strip())
        except Exception as e:
            continue

    df = pd.DataFrame({'datetime': dates, 'user': users, 'message': texts})
    df['user'] = df['user'].replace({'You': 'Me'})
    df['only_date'] = df['datetime'].dt.date
    df['year'] = df['datetime'].dt.year
    df['month_num'] = df['datetime'].dt.month
    df['month'] = df['datetime'].dt.strftime('%B')
    df['day_name'] = df['datetime'].dt.strftime('%A')
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['period'] = df['hour'].apply(hour_to_readable_period)

    return df
