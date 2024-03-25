from ultralytics import YOLO
import cv2
import os
import streamlit as st


class WebApp:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.create_sidebar()
        self.play_stored_video()

    def create_sidebar(self):
        with st.sidebar:
            self.show_original_video = st.radio(
                "Show original video", ["Yes", "No"], index=0, disabled=False
            )
            self.show_processed_video = st.radio(
                "Show processed video", ["Yes", "No"], index=1, disabled=False
            )

    def file_selector(self, folder_path="test-videos/"):
        file_names = os.listdir(folder_path)
        selected_file = st.selectbox("choose a file", file_names)
        return folder_path, selected_file

    def play_stored_video(self):

        folder_path, selected_file = self.file_selector()

        video_path = f"{folder_path}{selected_file}"

        if self.show_original_video == "Yes":
            st.markdown("# Original Video")
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()
            if video_bytes:
                st.video(video_bytes)
        if self.show_processed_video == "Yes":
            st.markdown("# Processed Video")
            try:
                vid_cap = cv2.VideoCapture(str(video_path))
                st_frame = st.empty()
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        self.display_detected_frames(
                            self.model,
                            st_frame,
                            image,
                        )
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error loading video: " + str(e))

    def display_detected_frames(
        self,
        model,
        st_frame,
        image,
        conf: float = 0.6,
    ):
        """
        Display the detected objects on a video frame using the YOLOv8 model.

        Args:
        - model (YoloV8): A YOLOv8 object detection model.
        - st_frame (Streamlit object): A Streamlit object to display the detected video.
        - image (numpy array): A numpy array representing the video frame.
        - conf (float): Confidence threshold for object detection.
        """

        # Resize the image to a standard size
        image = cv2.resize(image, (720, int(720 * (9 / 16))))

        res = model.predict(image, conf=conf)

        # # Plot the detected objects on the video frame
        res_plotted = res[0].plot()
        st_frame.image(
            res_plotted, caption="Detected Video", channels="BGR", use_column_width=True
        )


if __name__ == "__main__":
    webapp = WebApp()
