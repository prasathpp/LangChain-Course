import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import httpx

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
temperature = 0
api_type = "azure_ad"
request_timeout = 1000
api_key = os.getenv("TOKEN")

# -------------------------------
# Initialise LLM (your working format)
# -------------------------------
try:
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("OPENAI_GPT_DEPLOYMENT_NAME"),
        temperature=temperature,
        openai_api_version="2023-05-15",
        openai_api_type=api_type,
        azure_endpoint=os.getenv("OPENAI_CHAT_API_BASE"),
        openai_api_key=api_key,
        request_timeout=request_timeout,
        streaming=False,
        http_client=httpx.Client(verify=False, follow_redirects=True),
    )
except Exception as e:
    print("Error initialising the language model.")
    print(f"Please check your .env file and Azure credentials. Details: {e}")
    exit()

