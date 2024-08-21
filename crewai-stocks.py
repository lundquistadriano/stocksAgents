# IMPORT DAS LIBS
import json 
import os
import yfinance as yf
import streamlit as st
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

# CRIANDO YAHOO FINANCE TOOL 
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2022-08-08", end="2024-08-08")
    return stock

yahoo_finance_tool = Tool(
    name = "Ferramenta de finanças do Yahoo",
    description = "Obtém preços de ações para {ticket} dos últimos anos sobre uma empresa específica da API do Yahoo Finance",
    func = lambda ticket: fetch_stock_price(ticket)
)

# IMPORTANDO OPENAI LLM - GPT
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-4o-mini")

stockPriceAnalyst = Agent(
    role = "Análise de preço de ações sênior",
    goal = "Encontre o preço das ações {ticket} e analise tendências",
    backstory = """Você tem muita experiência em analisar o preço de uma ação específica e fazer previsões sobre seu preço futuro.""",
    verbose = True,
    llm = llm,
    max_iter = 5,
    memory = True,
    tools = [yahoo_finance_tool],
    allow_delegation = False
)

getStockPrice = Task(
    description = "Analisar o histórico de preços de ações {ticket} e criar uma análise de tendências de alta, baixa ou lateral",
    expected_output = """"Especifique a tendência atual do preço das ações - para cima, para baixo ou para os lados, por exemplo, stock= 'APPL, price UP'.""",
    agent = stockPriceAnalyst
)

# IMPORTANT A TOOL DE SEARCH 
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

newsAnalyst = Agent(
    role = "Analista de notícias de ações",
    goal = """Crie um breve resumo das notícias do mercado relacionadas à empresa de ações {ticket}. Especifique a tendência atual - para cima, para baixo ou para os lados com
o contexto das notícias. Para cada ativo de ação solicitado, especifique um número entre 0 e 100, onde 0 é medo extremo e 100 é ganância extrema.""",
    backstory = """Você tem muita experiência em analisar tendências e notícias de mercado e monitora ativos há mais de 10 anos.
    Você também é analista de nível mestre em mercados tradicionais e tem profundo conhecimento da psicologia humana.
    Você entende as notícias, os títulos e informações deles, mas olha para eles com uma dose saudável de ceticismo. 
    Considere também a fonte dos artigos de notícias. 
    """,
    verbose = True,
    llm = llm,
    max_iter = 10,
    memory = True,
    tools = [search_tool],
    allow_delegation = False
)

get_news = Task(
    description= f"""Pegue o estoque e sempre inclua BTC nele (se não for solicitado).
    Use a ferramenta de busca para pesquisar cada um individualmente. 

    A data atual é {datetime.now()}.

    Componha os resultados em um relatório útil""",
    expected_output = """"Um resumo do mercado geral e um resumo de uma frase para cada ativo solicitado. 
    Inclua uma pontuação de medo/ganância para cada ativo com base nas notícias. Use o formato:
    <ATIVO DE ESTOQUE>
    <RESUMO BASEADO EM NOTÍCIAS>
    <PREVISÃO DE TENDÊNCIAS>
    <PONTUAÇÃO DE MEDO/GANÂNCIA>
""",
    agent = newsAnalyst
)

stockAnalystWrite = Agent(
    role = "Escritor Analista Sênior de Ações",
    goal = """"Analise as tendências de preços e notícias e escreva um boletim informativo, envolvente e perspicaz, de 3 parágrafos, com base no relatório de ações e na tendência de preços. """,
    backstory = """Você é amplamente aceito como o melhor analista de ações do mercado. Você entende conceitos complexos e cria histórias e narrativas atraentes que ressoam com públicos mais amplos. 

    Você entende fatores macro e combina múltiplas teorias - por exemplo, teoria dos ciclos e análises fundamentais.
    Você é capaz de ter várias opiniões ao analisar qualquer coisa.
""",
    verbose = True,
    llm = llm,
    max_iter = 5,
    memory = True,
    allow_delegation = True
)

writeAnalyses = Task(
    description = """Use a tendência do preço das ações e o relatório de notícias sobre ações para criar uma análise e escrever um boletim informativo sobre a empresa {ticket} que seja breve e destaque os pontos mais importantes.
    Foco na tendência do preço das ações, notícias e pontuação de medo/ganância. Quais são as considerações para o futuro próximo?
    Inclua análises anteriores de tendências de ações e resumo de notícias.
""",
    expected_output = """"Um formato de boletim informativo eloquente de 3 parágrafos como markdown de uma maneira fácil de ler. Deve conter:

    - Resumo executivo de 3 marcadores
    - Introdução - define o quadro geral e desperta o interesse
    - a parte principal fornece o cerne da análise, incluindo o resumo das notícias e as pontuações de feed/greed
    - resumo - fatos principais e previsão concreta de tendências futuras - para cima, para baixo ou para os lados.
""",
    agent = stockAnalystWrite,
    context = [getStockPrice, get_news]
)

crew = Crew(
    agents = [stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks = [getStockPrice, get_news, writeAnalyses],
    verbose = True,
    process = Process.hierarchical,
    full_output = True,
    share_crew = False,
    manager_llm = llm,
    max_iter = 15
)

with st.sidebar:
    st.header('Insira o Stock para pesquisar')

    with st.form(key='research_form'):
        topic = st.text_input("Selecione o ticket")
        submit_button = st.form_submit_button(label = "Executar pesquisa")
if submit_button:
    if not topic:
        st.error("Por favor, preencha o campo de tíquete")
    else:
        results= crew.kickoff(inputs={'ticket': topic})

        st.subheader("Resultados da pesquisa:")
        final_result = results.tasks_output[-1].raw
        st.write(final_result)