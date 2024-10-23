from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
    openai_api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),
    azure_deployment=os.environ.get('AZURE_DEPLOYMENT_NAME'),
    openai_api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
)

print("-----Prompt from Template-----")
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic":"cats"})
result = model.invoke(prompt)
print(result.content)

print("\n-----Prompt with Multiple Placeholders-----\n")
template_multiple = """You are a hlepful assistant.
Huamn: Tell me a {adjective} story about a {animal}.
Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective":"funny", "animal":"panda"})
result = model.invoke(prompt)
print(result.content)

print("\n-----Prompt with System and Human Message (Tuple)-----\n")
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human","Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count":3})
result = model.invoke(prompt)
print(result.content)