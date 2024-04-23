import os
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
import nltk

def import_nltk():
    nltk.download('punkt')

def extract_video_id(url):
    # Parse the URL
    parsed_url = urlparse(url)
    if parsed_url.netloc == 'youtu.be':
        # The video ID is the path itself for 'youtu.be' URLs
        return parsed_url.path[1:]
    if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            # The video ID is in the 'v' query parameter for '/watch' paths
            return parse_qs(parsed_url.query)['v'][0]
        if parsed_url.path[:7] == '/embed/':
            # The video ID is the path itself for '/embed/' paths
            return parsed_url.path.split('/')[2]
        if parsed_url.path[:3] == '/v/':
            # The video ID is the path itself for '/v/' paths
            return parsed_url.path.split('/')[2]
    # Return None if no video ID could be extracted
    return None

def get_transcript_from_youtube(video_id: str):
    # nltk.download('punkt')
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = ' '.join([segment['text'] for segment in transcript_list])
    sentences = nltk.sent_tokenize(transcript)
    # context = ""
    os.makedirs("./data", exist_ok=True)
    with open(f"./data/{video_id}.txt", "w") as txt_file:
        for sentence in sentences:
            txt_file.write(sentence + '.\n')
            # context += sentence + '.\n'
    print(f"transcription of {video_id} was created")