import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY= os.getenv('OPENAI_API_KEY')
GROQ_API_KEY= os.getenv('GROQ_API_KEY')

api_wrap_arxiv = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
api_wrap_wiki= WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)

arxiv = ArxivQueryRun(api_wrapper=api_wrap_arxiv)
wiki = WikipediaQueryRun(api_wrapper=api_wrap_wiki)

search = DuckDuckGoSearchRun(name="search")

tools = [arxiv,wiki,search]

if "key_valid" not in st.session_state:
    st.session_state.key_valid = False
if "llm" not in st.session_state:
    st.session_state.llm = None

def check_key_model_match(key,model):

    if (key == OPENAI_API_KEY) and model in model_openai:
        st.session_state.llm = ChatOpenAI(model=model)
        return True
    if (key == GROQ_API_KEY) and model in model_groq:
        st.session_state.llm = ChatGroq(model=model)
        return True
    
    st.sidebar.write("Enter valid key for the model")
    return False

st.title("AI Agent example with Langchain")

api_key = st.sidebar.text_input("Enter your API Key",type="password")

model_openai=["gpt-3.5-turbo","gpt-4o"]
model_groq=["gemma2-9b-it","llama3-70b-8192","llama3-8b-8192"]
all_models = model_openai+model_groq

model = st.sidebar.selectbox("Choose your Model",all_models)

if check_key_model_match(api_key,model):
    st.session_state.key_valid = True
else:
    st.session_state.key_valid = False


if st.session_state.key_valid:

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt:= st.chat_input(placeholder="What is Deep Learning"):
        st.session_state.messages.append({"role":"user","content":prompt})
        st.chat_message("user").write(prompt)

        search_agent = initialize_agent(tools,llm=st.session_state.llm,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)

        with st.chat_message('assistant'):
            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write(response)
    
