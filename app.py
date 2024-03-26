from ultralytics import YOLO
import torch
import cv2
import os
import streamlit as st
from classes import classes as model_classes
import pandas as pd


class WebApp:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.initialize_session_state()
        self.create_sidebar()
        self.create_category_selector()
        # self.create_class_selector()
        # self.return_selected_classes()
        self.create_window()

    def initialize_session_state(self):
        if "clicked" not in st.session_state:
            st.session_state.clicked = True

    @st.cache_resource
    def load_model(_self):
        return YOLO("yolov8n.pt")

    @st.cache_data
    def load_classes(_self):
        df = pd.read_csv("classes.csv", index_col=0)
        return df

    def create_sidebar(self):
        with st.sidebar:
            self.inference_source = st.radio(
                "Inference Source", ["Video", "Webcam"], index=0, disabled=False
            )
            self.show_processed_video = st.radio(
                "Choose feed", ["Original", "Processed"], index=0, disabled=False
            )

            self.confidence = st.slider("Desired confidence:", 0.1, 1.0, 0.2, 0.1)
            self.iou = st.slider("Desired Intersection Over Union:", 0.1, 1.0, 0.7, 0.1)

    def file_selector(self, folder_path="test-videos/"):
        try:
            file_names = os.listdir(folder_path)
            selected_file = st.selectbox("choose a file", file_names)
            return folder_path, selected_file
        except:
            st.sidebar.error("Directory not found")

    def create_category_selector(self):
        with st.expander("select categories to be detected"):
            classes_df = self.load_classes()
            categories_df = pd.DataFrame(
                classes_df["category"].unique(), columns=["categories"]
            )
            categories_df["enabled"] = True

            data_editor = st.data_editor(
                categories_df,
                column_config={
                    "enabled": st.column_config.CheckboxColumn(
                        "enabled",
                        help="select the categories you want the model to detect",
                    ),
                },
                disabled=list(categories_df.columns[:-1]),
                hide_index=True,
                key="categories",
            )
            active_categories = data_editor[data_editor["enabled"] == True]
            active_categories = classes_df[
                classes_df["category"].isin(active_categories["categories"])
            ]
            self.selected_classes = list(active_categories.index.values)

    def create_class_selector(self):
        with st.expander("Select objects to be detected"):
            classes_df = self.load_classes()
            classes_df["enabled"] = False

            self.columns = list(classes_df.columns)

            self.categories = list(classes_df["category"].unique())

            self.filtered_df = {}

            for i, category in enumerate(self.categories):
                self.filtered_df[category] = classes_df[
                    classes_df["category"] == category
                ]
                self.filtered_df[category] = st.data_editor(
                    self.filtered_df[category],
                    column_config={
                        "enabled": st.column_config.CheckboxColumn(
                            "enabled",
                            help="Select the classes you want the model to detect",
                        )
                    },
                    disabled=list(classes_df.columns[:-1]),
                    hide_index=True,
                    key=category,
                )

    def return_selected_classes(self):

        total_df = pd.DataFrame(columns=self.columns)

        for category in self.categories:

            temp = self.filtered_df[category][
                self.filtered_df[category]["enabled"] == True
            ]
            total_df = pd.concat([total_df, temp])
        self.selected_classes = list(total_df.index.values)

    def display_detected_frames(
        self,
        model,
        st_frame,
        image,
    ):

        # Resize the image to a standard size
        image = cv2.resize(image, (720, int(720 * (9 / 16))))

        y_pred = model.predict(
            image,
            conf=self.confidence,
            stream_buffer=True,
            iou=self.iou,
            device=self.device,
            classes=self.selected_classes,
        )

        # # Plot the detected objects on the video frame
        boxes_plotted = y_pred[0].plot()
        st_frame.image(
            boxes_plotted,
            caption="Detected Video",
            channels="BGR",
            use_column_width=True,
        )

    def create_window(self):

        st.title("Object Detection Prototype")

        try:

            folder_path, selected_file = self.file_selector()

            video_path = f"{folder_path}{selected_file}"

            if self.show_processed_video != "Processed":
                st.markdown("# Original Video")

                with open(video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                if video_bytes:
                    st.video(video_bytes)

            if self.show_processed_video == "Processed":
                st.markdown("# Processed Video")

                try:
                    if self.inference_source == "Webcam":
                        video_feed = cv2.VideoCapture(0)
                    else:
                        video_feed = cv2.VideoCapture(str(video_path))

                    st_frame = st.empty()

                    while video_feed.isOpened():
                        success, image = video_feed.read()

                        if success:
                            self.display_detected_frames(
                                self.model,
                                st_frame,
                                image,
                            )
                        else:
                            video_feed.release()
                            break

                except Exception as e:
                    st.sidebar.error("Error loading video: " + str(e))

        except TypeError:
            st.markdown("we couldn't find")


if __name__ == "__main__":
    webapp = WebApp()
