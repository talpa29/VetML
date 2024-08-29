from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
import PyPDF2
import prompts

# load .env vars
_ = load_dotenv(find_dotenv())
client = OpenAI(
    api_key = os.environ.get('OPEN_AI_API_KEY')
)

# model = 'gpt-4-turbo-preview'
model='babbage-002'
temperature = 0.3 # less random, more deterministic (0.7 is default)
max_tokens = 500
topic = "" # what the summery topic is


# Read the pdf file
book = ""
file_path = "GOT.pdf"
# with open(file_path, 'rb') as file:
#     reader = PyPDF2.PdfReader(file)
#     for page in reader.pages:
#         book += page.extract_text()


# prompts
system_message = prompts.system_message
prompt = prompts.generate_prompt(book, topic)

messages = [
    {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
]
messages=[
        {"role": "system", "content": "You are a helpfull assistant"},
        {"role": "user", "content": "List 10 facts about dogs from the boxer breed"}
    ]

def get_summary():
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content

print(get_summary())

# completion = client.chat.completions.create(
#     model='gpt-3.5-turbo',
#     messages=[
#         {"role": "system", "content": "You are a poetice assistant"},
#         {"role": "user", "content": "Compose a poem that explain the love"}
#     ]
# 
# print(completion.choices[0].message)