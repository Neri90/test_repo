import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

class Settings:
    #os.getenv("OPENAI_API_KEY")
    LLM_CLIENT = client

    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = int(os.getenv('DB_PORT'))
    DB_USER = os.getenv('DB_USER')
    DB_PASS = os.getenv('DB_PASSWORD')
    DB_NAME = os.getenv('DB_NAME')
    DB_TABLE = os.getenv('DB_TABLE', 'ai_vector_store')
    #                                       └> 만약 db_table이 없으면 ai_vector_store 넣어라

Settings = Settings()
