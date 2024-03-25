import streamlit as st
from streamlit_player import st_player
import os

st.title("Object detection prototype")


def file_selector(folder_path="test-videos/"):
    file_names = os.listdir(folder_path)
    selected_file = st.selectbox("choose a file", file_names)
    return f"{folder_path}{selected_file}"


user_input = file_selector()

if user_input is not None:
    st.markdown("# Original Video")
    video_file = open(user_input, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)
