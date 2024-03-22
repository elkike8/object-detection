from pytube import YouTube

link = "https://www.youtube.com/watch?v=_7RfMf8FLXY"

YouTube(link).streams.first().download()
yt = YouTube(link)

yt.streams.filter(progressive=True, file_extension="mp4").order_by(
    "resolution"
).desc().first().download()
