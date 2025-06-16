import logging
import os
import ssl
from typing import Any

import certifi
import weave
from dotenv import load_dotenv
from langchain_core.messages.utils import (
    count_tokens_approximately,
)
from langchain_openai import ChatOpenAI

# Define the set of web tools our agent will use to interact with the Tavily API.
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langmem.short_term import SummarizationNode

from backend.prompts import PROMPT

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class State(AgentState):
    # NOTE: we're adding this key to keep track of previous summary information
    # to make sure we're not summarizing on every LLM call
    context: dict[str, Any]


# --- Agent Class Definition ---
class WebAgent:
    """
    Template for a LangGraph agent that routes between LLM and web search.
    """

    def __init__(self):
        # Fix SSL context for FastAPI/uvicorn environment
        self._setup_ssl_context()

        # 3a. Initialization of LLMs, Chains, and Clients
        self.o3_mini = ChatOpenAI(
            model="o3-mini-2025-01-31", api_key=os.getenv("OPENAI_API_KEY")
        )
        self.gpt_4_1 = ChatOpenAI(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))
        self.gpt_4_1_nano = ChatOpenAI(
            model="gpt-4.1-nano", api_key=os.getenv("OPENAI_API_KEY")
        )

        self.streaming_llm = self.gpt_4_1_nano.with_config({"tags": ["streaming"]})

        # Define the LangChain search tool
        self.search = TavilySearch(
            max_results=10, topic="general", api_key=os.getenv("TAVILY_API_KEY")
        )

        # Define the LangChain extract tool
        self.extract = TavilyExtract(
            extract_depth="advanced", api_key=os.getenv("TAVILY_API_KEY")
        )
        self.checkpointer = MemorySaver()
        # Define the LangChain crawl tool
        self.crawl = TavilyCrawl(api_key=os.getenv("TAVILY_API_KEY"))
        self.weave_client = weave.init("tavily-demo")
        self.prompt = PROMPT
        # Tavily web search client (API key should be set in environment)
        # self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.summarization_model = ChatOpenAI(model="gpt-4.1-nano").bind(max_tokens=500)

        self.summarization_node = SummarizationNode(
            token_counter=count_tokens_approximately,
            model=self.summarization_model,
            max_tokens=384,
            max_summary_tokens=128,
            output_messages_key="llm_input_messages",
        )

    def _setup_ssl_context(self):
        """Setup SSL context to fix certificate verification issues in FastAPI/uvicorn."""
        try:
            # Set SSL certificate file path
            os.environ["SSL_CERT_FILE"] = certifi.where()
            os.environ["SSL_CERT_DIR"] = certifi.where()

            # Create default SSL context with proper certificates
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            ssl._create_default_https_context = lambda: ssl_context

        except Exception as e:
            logger.warning(f"Could not setup SSL context: {e}")

    # 3c. Define the Flow of the Graph
    def build_graph(self):
        """
        Build and compile the LangGraph workflow.
        """
        return create_react_agent(
            prompt=self.prompt,
            model=self.streaming_llm,
            tools=[self.search, self.extract, self.crawl],
            # pre_model_hook=self.summarization_node,
            checkpointer=self.checkpointer,
        )


# --- Example Usage (for reference) ---
if __name__ == "__main__":
    agent = WebAgent()
    compiled_agent = agent.build_graph()
    # Example state
    from langchain.schema import HumanMessage

    # Test the web agent
    inputs = {"messages": [HumanMessage(content="who is the ceo of tavily?")]}
    # Stream the web agent's response
    weave.op()
    for s in compiled_agent.stream(inputs, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
