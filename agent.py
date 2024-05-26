from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_community.tools.shell.tool import ShellTool
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_community.document_loaders import DirectoryLoader
from typing import Dict, List, Optional
from dotenv import load_dotenv
from utils import *
import subprocess
import json
import os


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "code-review-bot"


@tool
def get_staged_changes_diffs(filename: str) -> Optional[str]:
    """
    Fetches the current diff for a file or returns None for error

    Parameters:
    filename (str): The filename to fetch git diffs for

    Returns:
    (Optional[str]): The list of diffs for a given file or None if they cannot be fetched
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--staged"], capture_output=True, text=True
        )

        if result.returncode == 0:
            return result.stdout
        else:
            return None
    except Exception as e:
        return None


@tool
def get_changed_diffs(filename: str) -> Optional[str]:
    """
    Fetches the current diff for a file or returns None for error

    Parameters:
    filename (str): The filename to fetch git diffs for

    Returns:
    (Optional[str]): The list of diffs for a given file or None if they cannot be fetched
    """
    try:
        result = subprocess.run(["git", "diff"], capture_output=True, text=True)

        if result.returncode == 0:
            return result.stdout
        else:
            return None
    except Exception as e:
        return None


@tool
def get_codebase_context(search: str) -> List[str]:
    """
    Takes in a search returns similar documents from a vector database

    Parameters:
    search (str): A search to embed and perform a vector database search for

    Returns:
    (List[str]): The list of similar document founds in the vector database
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    document_vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, embedding=embeddings
    )

    retriever = document_vectorstore.as_retriever()

    return retriever.invoke(search)


# First, embed entire codebase
embed_documents(INDEX_NAME)

tools = [
    ShellTool(ask_human_input=True),
    get_staged_changes_diffs,
    get_codebase_context,
    get_changed_diffs,
]

# Create LLM with tools to perform code review
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Set up the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert web developer skilled in code reviewing. When the user asks a question:

            1. First, fetch the current diffs.
            2. If the information is not found in the diffs, use the codebase context.

            Keep your answers very short with no yapping. Do not ask which files to check; parse through the diffs and codebase context to determine that yourself. Format your response for a terminal with multiple line breaks before and after.
            """,
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Bind the tools to the language model
llm_with_tools = llm.bind_tools(tools)

# Create the agent
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Main loop to prompt the user
while True:
    user_prompt = input("Prompt: ")
    list(agent_executor.stream({"input": user_prompt}))
