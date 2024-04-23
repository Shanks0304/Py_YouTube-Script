from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from pinecone import Pinecone as Pinecone_Init 
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
import tiktoken
import time


load_dotenv()
tokenizer = tiktoken.get_encoding('cl100k_base')

pc = Pinecone_Init(
    api_key=os.getenv('PINECONE_API_KEY')
)

index_name = os.getenv('PINECONE_INDEX')
embeddings = OpenAIEmbeddings()


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)



def split_document(doc: Document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents([doc])
    return chunks

def create_index():
    print("We are generating new database for training GPT model.")
    pc.create_index(
        name=index_name, 
        dimension=1536, 
        metric="cosine",
        spec= ServerlessSpec(
            cloud='aws',
            region= 'us-east-1'
        ))
    print("We can create a new database.")
    
def initialize_pinecone():
    # print(pc.list_indexes())
    if len(pc.list_indexes()):
        print("Previous training data will be removed")
        is_delete_index = input("Do you agree with it?(y/n): ")
        if is_delete_index == 'y':
            pc.delete_index(name=index_name)
            print("Previous training data is removed successfully, Congratulations!")
            create_index()
            
        else:
            print("The response will be generated based on both past and present training data.")
    else:
        create_index()

def train_txt(filename: str):
    start_time = time.time()
    loader = TextLoader(file_path=f"./data/{filename}")
    documents = loader.load()
    total_content = ""
    for document in documents:
        total_content += "\n\n" + document.page_content
    doc = Document(page_content=total_content, metadata={"source": filename})
    # print(filename)

    chunks = split_document(doc)
    
    Pinecone.from_documents(
        chunks, embeddings, index_name=index_name)
    
    end_time = time.time()
    print(f"{filename} is trained on GPT.")
    print("training time: ", end_time - start_time)
    return True

def get_context(msg: str, number: int):
    # print("message: " + msg)
    similarity_value_limit = 0.6

    results = tuple()
    db = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings)

    results = db.similarity_search_with_score(msg, k=number)
    # print(results)
    
    context = ""
    
    for result in results:
        if result[1] >= similarity_value_limit:
            context += f"\n\n{result[0].page_content}"
    
    return context
