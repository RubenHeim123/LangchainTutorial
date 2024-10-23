from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

print("-----Prompt from Template-----")
prompt = prompt_template.invoke({"topic":"cats"})
print(prompt)

template_multiple = """You are a hlepful assistant.
Huamn: Tell me a {adjective} story about a {animal}.
Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective":"funny", "animal":"panda"})
print("\n----- Prompt with Multiple Placeholders -----\n")
print(prompt)

messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me 3 jokes")
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers"})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)



# model = AzureChatOpenAI(
#     azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
#     openai_api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),
#     azure_deployment=os.environ.get('AZURE_DEPLOYMENT_NAME'),
#     openai_api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
# )
