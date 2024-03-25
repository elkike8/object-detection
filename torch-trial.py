# https://pytorch.org/hub/ultralytics_yolov5/
# https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/
# https://towardsdatascience.com/implementing-real-time-object-detection-system-using-pytorch-and-opencv-70bac41148f7

# import cv2  # opencv2 package for python.
# import pafy  # pafy allows us to read videos from youtube.URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" #URL to parse

# play = pafy.new(self._URL).streams[-1]  #'-1' means read the lowest quality of video.
# assert play is not None  # we want to make sure their is a input to read.
# stream = cv2.VideoCapture(play.url)  # create a opencv video stream.

# read from local camera.
# stream = cv2.VideoCapture(0)

# read from ip camera
# camera_ip = "rtsp://username:password@IP/port"
# stream = cv2.VideoCapture(camera_ip)

import torch
import numpy as np
import cv2
import pafy
import pytube
from time import time


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a video using Opencv2.
    """

    def __init__(self):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_video_stream(self, path_to_video):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        # play = pafy.new(self._URL).streams[-1]
        # assert play is not None
        # return cv2.VideoCapture(play.url)
        video_name = path_to_video[
            path_to_video.rfind("/") + 1 : path_to_video.rfind(".")
        ]
        video_path = path_to_video[: path_to_video.rfind("/") + 1]

        return video_path, video_name, cv2.VideoCapture(path_to_video)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = (
                    int(row[0] * x_shape),
                    int(row[1] * y_shape),
                    int(row[2] * x_shape),
                    int(row[3] * y_shape),
                )
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(
                    frame,
                    self.class_to_label(labels[i]),
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    bgr,
                    2,
                )

        return frame

    def process_video(self, path_to_video):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        video_path, video_name, player = self.get_video_stream(path_to_video)
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            f"processed-videos/{video_name}.avi", four_cc, 20, (x_shape, y_shape)
        )
        while True:
            start_time = time()
            ret, frame = player.read()
            assert ret
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 3)
            print(f"Frames Per Second : {fps}")
            out.write(frame)


# Create a new object and execute.

if __name__ == "__main__":

    video_1 = "https://www.youtube.com/watch?v=PKYriLfSXqg"
    video_2 = "https://www.youtube.com/watch?v=_7RfMf8FLXY"
    video_3 = "https://www.youtube.com/watch?v=Cszy1AwhB4Y"
    local_path = "test-videos/man-dog-fence.mp4"

    instance = ObjectDetection()
    instance.process_video(local_path)
