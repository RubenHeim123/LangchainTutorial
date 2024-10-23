from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
import os

load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
    openai_api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),
    azure_deployment=os.environ.get('AZURE_DEPLOYMENT_NAME'),
    openai_api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
)

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Your are a helpful assistant."),
        ("human", "Generate a thank you note for this positive feedback {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Your are a helpful assistant."),
        ("human", "Generate a response addressing this negative feedback {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Your are a helpful assistant."),
        ("human", "Generate a request for more details for this neutral feedback {feedback}."),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Your are a helpful assistant."),
        ("human", "Generate a message to escalate this feedback to a human agent: {feedback}."),
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Your are a helpful assistant."),
        ("human", "Classify the sentiment of this feedback as positive, negative, neutral or escanlate: {feedback}."),
    ]
)

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain | branches

# Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I´m not sure about the product yet. Can you tell me more about its features and benefits?"

review = "I´m not sure about the product yet. Can you tell me more about its features and benefits?"
result = chain.invoke({"feedback": review})

print(result)