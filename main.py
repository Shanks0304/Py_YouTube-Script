from app.utils.chatgpt import gpt_response
from app.utils.youtube_transcript import extract_video_id, get_transcript_from_youtube, import_nltk 
from app.utils.pinecone import initialize_pinecone, train_txt, get_context


def main():
    # Open the input txt file
    with open('input.txt', 'r') as file:
        # Read the lines
        lines = file.readlines()
    

    # Initialize an empty list to store the links
    video_ids = []
    
    # Loop over the lines
    import_nltk()
    initialize_pinecone()
    print("\nWe can start extracting transcriptions of YouTube videos")
    for line in lines:
        video_id = extract_video_id(line.strip())
        video_ids.append(extract_video_id(line.strip()))
        get_transcript_from_youtube(video_id=video_id)
        train_txt(f"{video_id}.txt")
    #     print(video_id)
    print("Congratulations! Training GPT model is finished, You can start now!")
    # Input the user query
    while(True):
        input_context = input("Please enter the question you'd like to ask!\n")
        # Semantic search from input text
        reference_context = get_context(input_context, len(lines))
        print("\nTrained GPT is ready to work.")
        # print(reference_context)
        print("-------------------------------------------------------------------------------------------\n\n")
        print("GPT is working now!")
        isSuccess = gpt_response(
            input_context=input_context,
            reference_context=reference_context
        )
        if not isSuccess:
            break
        
        isContinue = input("Thanks for using this app!, Do you want to continue to use this app? (y/n): ")
        if isContinue == 'n':
            break
        if isContinue == 'y':
            continue


if __name__ == "__main__":
    main()