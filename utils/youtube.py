from pytube import YouTube

link = "https://www.youtube.com/watch?v=_7RfMf8FLXY"


def main(link: str):
    """Downloads a video from youtube to current working directory.

    Args:
        link (str): Url for the video to download
    """
    YouTube(link).streams.first().download()
    yt = YouTube(link)

    yt.streams.filter(progressive=True, file_extension="mp4").order_by(
        "resolution"
    ).desc().first().download()


if __name__ == "__main__":
    main(link=link)
