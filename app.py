from ultralytics import YOLO
import torch
import cv2
import os
import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple


class WebApp:
    def __init__(self):
        self.page_setup()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.create_sidebar()
        self.create_category_selector()
        self.create_main_window()

    @st.cache_resource
    def load_model(_self):
        """Loads a pretrained object detection model. If first time using, it downloads it, else loads from memory.

        Args:
            _self (_type_): _description_

        Returns:
            model: A pretrained object detection model.
        """
        return YOLO("yolov8n.pt")

    @st.cache_data
    def load_classes(_self):
        """Loads a csv containing the classes the object detection model recognizes.

        Args:
            _self (_type_): _description_

        Returns:
            dataframe: A dataframe containing the classes the model recognizes.
        """
        df = pd.read_csv("classes.csv", index_col=0)
        return df

    def page_setup(self):
        """Formats the streamlit app page."""
        title = "Object Detection Prototype"
        st.set_page_config(
            page_title=title,
            page_icon=":movie_camera:",
            layout="wide",
        )
        st.title(title)

    def create_sidebar(self):
        """Created a side bar for the streamlit app. The sidebar contains variables that send arguments to the object detection model."""
        with st.sidebar:
            # select from a video or a webcam feed
            self.inference_source = st.radio(
                "Inference Source", ["Video", "Webcam"], index=0, disabled=False
            )
            # select to display the original video or the processed "real time" images from the model
            self.show_processed_video = st.radio(
                "Choose feed", ["Original", "Processed"], index=0, disabled=False
            )
            # lowest confidence the boxes will be displayed
            self.confidence = st.slider("Desired confidence:", 0.1, 1.0, 0.2, 0.1)
            # the higher the iou the less likely boxes will appear over overlapped objects
            # higher values also help avoiding double boxes over single objects
            self.iou = st.slider("Desired Intersection Over Union:", 0.1, 1.0, 0.7, 0.1)

    def create_category_selector(self):
        """Creates a streamlit element that allows the user to select which object categories will they like the model to detect."""
        with st.expander("Select the object you want to be detected"):
            # creating two columns, one to place the category selector one for a display showing the selected classes
            col1, col2 = st.columns(2)

            # configuration for the category selector
            col1.header("Select categories to detect")
            with col1:

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

            # configuration for the display
            col2.header("Selected classes to be detected")

            with col2:
                st.data_editor(
                    active_categories.sort_values(by=["category"]),
                    key="other",
                    disabled=list(active_categories.columns),
                    hide_index=True,
                )

    def file_selector(self, folder_path="test-videos/"):
        """Creates a stremlit element to choose the video to process from a local directory.

        Args:
            folder_path (str, optional): local path where video is stored. Defaults to "test-videos/".

        Returns:
            _type_: _description_
        """
        try:
            file_names = os.listdir(folder_path)
            selected_file = st.selectbox("choose a file", file_names)
            return folder_path, selected_file
        except:
            st.sidebar.error("Directory not found")

    def display_detected_frames(
        self,
        model,
        st_frame,
        image,
    ):
        """Uses the model to detect objects over frames of the original video. Display the detected boxes over the processed image.

        Args:
            model (_type_): the pretrained object detection model.
            st_frame (_type_): streamlit object to display the processed image
            image (_type_): the current frame of the video to be processed
        """
        # Resize (and pad if necessary) the image to a standard size
        # according to the docs, best results are achieved when the image size is divisible by 64 since if fits properly into the convolution layers
        image = self.resize_image_with_pad(image, (960, 960))

        # predict the objects in the image
        y_pred = model.predict(
            image,
            conf=self.confidence,
            stream_buffer=True,
            iou=self.iou,
            device=self.device,
            classes=self.selected_classes,
        )

        # plot the detected objects over the streamlit object
        boxes_plotted = y_pred[0].plot()
        st_frame.image(
            boxes_plotted,
            caption="Detected Video",
            channels="BGR",
            use_column_width=True,
        )

    def create_main_window(self):
        """Places the main visual elements on the streamlit app."""
        try:
            folder_path, selected_file = self.file_selector()

            video_path = f"{folder_path}{selected_file}"

            # displays original video
            if self.show_processed_video != "Processed":
                st.markdown("# Original Video")

                with open(video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                if video_bytes:
                    st.video(video_bytes)

            # displays the processed images continuously so they resemble a video
            if self.show_processed_video == "Processed":
                st.markdown("# Processed Video")

                try:
                    if self.inference_source == "Webcam":
                        video_feed = cv2.VideoCapture(0)
                    else:
                        video_feed = cv2.VideoCapture(str(video_path))

                    # streamlit object to display processed images
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

    def resize_image_with_pad(
        self,
        image: np.array,
        new_shape: Tuple[int, int],
        padding_color: Tuple[int] = (255, 255, 255),
    ) -> np.array:
        """Resizes and pads (if necessary) an image to a desired size.

        Args:
            image (np.array): image to be processed
            new_shape (Tuple[int, int]): desired output size of the image
            padding_color (Tuple[int], optional): RGB for the color tu use as padding. Defaults to white (255, 255, 255).

        Returns:
            np.array: resized and padded image
        """
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(new_shape)) / max(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])

        if new_size[0] > new_shape[0] or new_size[1] > new_shape[1]:
            ratio = float(min(new_shape)) / min(original_shape)
            new_size = tuple([int(x * ratio) for x in original_shape])

        image = cv2.resize(image, new_size)
        delta_w = new_shape[0] - new_size[0]
        delta_h = new_shape[1] - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            None,
            value=padding_color,
        )
        return image


if __name__ == "__main__":
    webapp = WebApp()
