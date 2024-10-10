from typing import Any

from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun


load_dotenv()


def main():

    print("*****************************************************")

    instructions = """You are an agent designed to answer questions using Yahoo Finance News Tool and Wikipedia.
              Your primary task is to provide the latest financial news and insights using the Yahoo Finance News Tool.
              You can also fetch general knowledge and non-financial information using the Wikipedia tool.
              Prioritize using the Yahoo Finance News Tool for finance-related queries.
              For non-finance queries, use the Wikipedia tool to provide accurate information.
              If neither tool can provide relevant data, respond with "I don't know".
              """

    wikipedia_api_wrapper = WikipediaAPIWrapper()
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [YahooFinanceNewsTool(), wikipedia_tool]

    agent_finance = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )

    agent_finance_executor = AgentExecutor(
        agent=agent_finance, tools=tools, verbose=True
    )

    agent_wiki = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )

    agent_wiki_executor = AgentExecutor(agent=agent_wiki, tools=tools, verbose=True)

    # router grand agent

    def agent_finance_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return agent_finance_executor.invoke({"input": original_prompt})

    def agent_wiki_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return agent_wiki_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name="Yahoo Finance Agent",
            func=agent_finance_executor_wrapper,
            description="Provides the latest financial news and insights using the Yahoo Finance News Tool",
        ),
        Tool(
            name="Wikipedia agent",
            func=agent_wiki_executor_wrapper,
            description="Fetches general knowledge and non-financial information using the Wikipedia tool.",
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )

    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    print(
        grand_agent_executor.invoke(
            {"input":  "What happens today with Microsoft stocks?",}
        )
    )


if __name__ == "__main__":
    main()
