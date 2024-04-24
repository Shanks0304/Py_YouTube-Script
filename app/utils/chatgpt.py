from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()
client = OpenAI()


def gpt_response(input_context: str, reference_context: str):
    # Step 1: send the conversation and available functions to GPT
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model='gpt-4-0125-preview',
            max_tokens=2000,
            messages=[
                {'role': 'system', 'content': "Please response the proper result."},
                {'role': 'user', 'content': f"""
                    This is the input question I'd like to know.
                    {input_context}
                    and this is the text you should reference: 
                    {reference_context}
                    
                    That's all.
                    You should create the proper answer for providing input question from the reference context 
                """}
            ],
            seed= 6601,
            temperature = 0.5,
            stream=True,
        )
        # response_message = response.choices[0].message.content
        # system_fingerprint = response.system_fingerprint
        # print("respons: ", response_message)
        for chunk in response:
             if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\nElapsed Time: ", time.time() - start_time)
        return True
    except Exception as e:
        print(e)
        return False