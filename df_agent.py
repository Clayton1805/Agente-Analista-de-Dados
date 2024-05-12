from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain_openai import ChatOpenAI
import pandas as pd

# from langchain.agents.agent_types import AgentType

# from langchain_community.llms import Ollama
# from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# llm = Ollama(model="llama3:8b")
chat_llm = ChatOllama(model="llama3:70b", temperature=0)
# print(llm.invoke("What are the capitals of Latin American countries?"))
# text="Quais são as capitais dos países latino-americanos?"


# template = "Você é um assistente útil que responde somente em {output_language}."
template = """
Você se chama Gilson, e está trabalhando com dataframe pandas no Python.
O nome do Dataframe é `df`.
"""
# chat_prompt = ChatPromptTemplate.from_messages([
#     ("system", template),
#     ("human", "{input}"),
# ])
# print(chat_prompt.format_messages(output_language="português"))


# print(chain.invoke())
# print(chain.invoke({ "input": "Quais são as capitais dos países latino-americanos?", "output_language": "português" }))


df = pd.read_csv("df_rent.csv")


agent = create_pandas_dataframe_agent(
    chat_llm,
    df,
    verbose=True,
    # prefix=template,
    # agent_type=AgentType.OPENAI_FUNCTIONS,
)
print(agent.invoke("Qual o tamanho do conjunto de dados que lhe passei?"))
# chain = agent | chat_prompt | chat_llm
# print(chain.invoke({ "input": "Qual o tamanho do conjunto de dados que lhe passei?", "output_language": "português" }))