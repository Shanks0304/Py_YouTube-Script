from youtube_transcript import extract_video_id, get_transcript_from_youtube
def main():
    # Open the input txt file
    with open('input.txt', 'r') as file:
        # Read the lines
        lines = file.readlines()
    # Initialize an empty list to store the links
    contexts = []
    #Loop over the lines
    for idx, line in enumerate(lines):
        video_id = extract_video_id(line.strip())
        contexts.append(get_transcript_from_youtube(video_id=video_id, idx=idx))
        print(video_id)
    return contexts

if __name__ == "__main__":
    main()