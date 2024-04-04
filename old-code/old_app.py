import streamlit as st
import os
from object_detector import ObjectDetector
import time


def file_selector(folder_path="test-videos/"):
    file_names = os.listdir(folder_path)
    selected_file = st.selectbox("choose a file", file_names)
    return folder_path, selected_file


def create_app(model):
    st.title("Object detection prototype")
    folder_path, selected_file = file_selector()

    if selected_file is not None:
        path_to_video = f"{folder_path}{selected_file}"
        st.markdown("# Original Video")
        video_file = open(path_to_video, "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)
        model.process_video(path_to_video)
        processed_file = open(f"processed-videos/{selected_file}", "rb")
        processed_bytes = processed_file.read()
        st.video(processed_bytes)


if __name__ == "__main__":

    detector = ObjectDetector()
    create_app(detector)
