import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

@st.cache_data(show_spinner=False)
def load_chat(data):
    return preprocessor.preprocess(data)

# ----------------------- SIDEBAR -----------------------
st.sidebar.title("📱 WhatsApp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("📄 Upload WhatsApp Chat File")
show_user_media_stats = st.sidebar.checkbox("📊 Show Per-User Media Stats")

# ----------------------- CHAT PROCESSING -----------------------
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = load_chat(data)

    # ✅ Media summary
    summary_df = helper.get_unique_media_summary_from_chat(df)
    if not summary_df.empty:
        st.sidebar.markdown("### 📊 Unique Media Summary (From Chat)")
        st.sidebar.dataframe(summary_df.set_index("Media Type"))
    else:
        st.sidebar.info("No media references (IMG-/VID-/PTT-/STK-) found in the chat.")

    # ✅ Chat type
    valid_users = df['user'].unique().tolist()
    if 'group_notification' in valid_users:
        valid_users.remove('group_notification')
    valid_users = [u for u in valid_users if u.strip() != '']
    chat_type = "Group Chat" if len(valid_users) > 2 else "1-to-1 Chat"
    st.sidebar.markdown(f"🧾 **Chat Type:** {chat_type}")

    # ✅ Media omitted count
    media_omitted = helper.count_media_placeholders(df)
    st.sidebar.metric("🔒 Media Messages (Omitted)", media_omitted)

    # ✅ Per-user media stats
    if show_user_media_stats:
        media_df = helper.get_user_media_breakdown(df)
        if not media_df.empty:
            st.title("📊 Per-User Media Insights")
            st.subheader("📢 Most Media-Active Users")
            st.dataframe(media_df['user'].value_counts().reset_index().rename(columns={"index": "User", "user": "Media Count"}))

            st.subheader("📁 Most Shared Media Types")
            st.bar_chart(media_df['type'].value_counts())

            st.subheader("📅 Average Media Sent Per Day")
            avg_per_day = media_df.groupby('date').size().mean()
            st.metric("Average Per Day", f"{avg_per_day:.2f}")

            st.subheader("🗓️ Media Usage Heatmap")
            heatmap_data = media_df.pivot_table(index='day', columns='period', aggfunc='size', fill_value=0)
            fig, ax = plt.subplots()
            sns.heatmap(heatmap_data, cmap='YlGnBu', ax=ax)
            st.pyplot(fig)
        else:
            st.title("📭 No Media Filenames Detected")
            st.warning("⚠️ No media filename references (IMG-/VID-/PTT-/STK-) found in chat messages.")

    # ✅ Media Type Breakdown Per User
    st.subheader("📸 Media Type Breakdown Per User")
    breakdown_df = helper.get_user_media_type_counts_from_chat(df)
    if not breakdown_df.empty:
        st.dataframe(breakdown_df)
        desired_columns = ['Images', 'Videos', 'Audio', 'Stickers']
        existing_columns = [col for col in desired_columns if col in breakdown_df.columns]

        if existing_columns:
            st.subheader("📊 Media Type by Top Users")
            top_users = breakdown_df.sort_values(by=existing_columns, ascending=False).head(5)
            melted = top_users.melt(id_vars='User', value_vars=existing_columns, var_name='Media Type', value_name='Count')
            fig = px.bar(melted, x='User', y='Count', color='Media Type', barmode='group')
            st.plotly_chart(fig)
        else:
            st.info("No media types (IMG-/VID-/PTT-/STK-) found to display.")
    else:
        st.info("No media references found in chat file.")

    # ✅ User selection
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list = [user for user in user_list if user.strip() != ""]
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("📌 Show analysis wrt", user_list)

    # ✅ Show analysis button
    if st.sidebar.button("Show Analysis"):
        # ✅ Top Stats
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", num_messages)
        col2.metric("Total Words", words)
        col3.metric("Media Shared", num_media_messages)
        col4.metric("Links Shared", num_links)

        # ✅ Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # ✅ Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # ✅ Activity maps
        st.title('Activity Map')
        col1, col2 = st.columns(2)
        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # ✅ Weekly heatmap
        st.title("Weekly Activity Heatmap")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        if not user_heatmap.empty:
            fig, ax = plt.subplots()
            sns.heatmap(user_heatmap, cmap='YlOrBr', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough data to generate heatmap.")

        # ✅ Most busy users
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # ✅ WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        ax.axis("off")
        st.pyplot(fig)

        # ✅ Most common words
        st.title('Most Common Words')
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # ✅ Emoji analysis
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            if not emoji_df.empty:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                st.pyplot(fig)
            else:
                st.warning("No emojis found in selected chat.")

    # ✅ Export cleaned data
    st.subheader("⬇️ Export")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Cleaned Chat Data", csv, "chat_data.csv", "text/csv")
