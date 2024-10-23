from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os

load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
    openai_api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),
    azure_deployment=os.environ.get('AZURE_DEPLOYMENT_NAME'),
    openai_api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
)

result = model.invoke("What is 81 divided by 9?")
print("Full result:")
print(result)
print("Content only")
print(result.content)
