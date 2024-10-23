from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os

load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
    openai_api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),
    azure_deployment=os.environ.get('AZURE_DEPLOYMENT_NAME'),
    openai_api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Your are a comdeian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"topic": "lawyers", "joke_count":3})

print(result)
