import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import shutil

#Loading of OpenAI API key
key = st.secrets["api_key"]

#General configuration of page
st.set_page_config(page_title = 'Coester AI PLC Program Verifier', layout = 'wide', page_icon = 'android-chrome-192x192.png')

hide_decoration_bar_style = ''' <style> header {visibility: hidden;} </style> '''

st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

st.title('Coester AI PLC Program Verifier', anchor = False)
st.write('AI Agent built using as reference the CENELEC Standards')
st.write('Version 1.0')

if st.button('Clear Chat History'):
    st.session_state['page_refresh'] = True

if 'page_refresh' in st.session_state:
    st.session_state.clear()
    st.rerun()

st.divider()

st.sidebar.image('coester_azul-01_-_sem fundo.png', width= 200)
st.sidebar.title('Configuration:')

#Entry of files
uploaded_files = st.sidebar.file_uploader('Load the code files (.xml) for verification:', type = ['xml'], accept_multiple_files = True)

#Selection of program language
language = st.sidebar.radio('Select the PLC Program Language:',
                            ['Ladder', 'FBD', 'STL', 'SCL'])

#Description of program subject
subject = st.sidebar.text_input('Insert the program subject:\n\n\nExample: valve control')


#Management of uploaded files
if uploaded_files:
        
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok = True)

    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())

    xml_files = [file for file in os.listdir(temp_dir)]

    all_codes = []

    for idx, (xml_file) in enumerate(xml_files, start=1):  # Começa a contagem do page em 1
        xml_path = os.path.join(temp_dir, xml_file)
        
        with open(xml_path, 'rb') as file:
            code = BeautifulSoup(file, 'xml').prettify()

        # Criando o objeto Document com o page iterado
        doc = Document(
            metadata={'source': xml_path, 'page': idx, 'page_label': str(idx)},
            page_content=code
        )

        all_codes.append(doc)        
            
    shutil.rmtree(temp_dir)

    if all_codes:
        st.sidebar.success('Files uploaded successfully.')
    
    #Loading of Embeddings model and LLM model
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small', openai_api_key = key)
    model = ChatOpenAI(model_name = 'o3-mini', api_key = key, temperature = 0)

    #Configuration of splitter
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Tamanho das partes
    chunk_overlap=100,  # Sobreposição entre partes
    separators=["\n\n", "\n"]
    )
    
    splits = text_splitter.split_documents(all_codes)
    
    #Generation of vectorstore of codes
    vectorstore = FAISS.from_documents(splits, embeddings)    

    #Configuration of retriever
    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 50, 'fetch_k': 100, 'lambda_mult': 0.25}
    )

    #Prompts of each program language
    prompt_ladder = f""" 
    You are an expert to verify PLC programs in Ladder.

    Your primary objective is to ensure the safety, reliability, and proper functionality of software used to an Automated People Mover, to be certified SIL 4 as per CENELEC standards.

    Your responses must be:
    - Clear, precise, and technically detailed.
    - Aligned with automated people mover standards and including EN 50128 as references.
            
    **Guidelines for Analysis**:
    - **Safety Priority**: Under no circumstances should you suggest modifications or enhancements that violate established safety principles, even if a requirement is found to be unmet.
    - **Thoroughness**: Analyze the program step-by-step to ensure a comprehensive understanding of its logic, structure, and functionality. Consider all the names of inputs, outputs, auxiliares, InOut, Temp, Return, Static, Network Names, Constants and comments to enhance your interpretation.
    - **Clarity**: If any part of the code or requirements is unclear or incomplete, specify what additional information is needed.

    The xml files are related to the control of {subject}. Receive the files and wait for the queries.

    For the complementary questions, after check of requirement, only reply directly about the additional question.
    
    xml files: {{snippets}}

    Query: {{query}}

    Memory: {{memory}}
    """

    prompt_stl = f""" 
    You are an expert to verify PLC programs in STL.

    Your primary objective is to ensure the safety, reliability, and proper functionality of software used to an Automated People Mover, to be certified SIL 4 as per CENELEC standards.

    Your responses must be:
    - Clear, precise, and technically detailed.
    - Aligned with automated people mover standards and including EN 50128 as references.
            
    **Guidelines for Analysis**:
    - **Safety Priority**: Under no circumstances should you suggest modifications or enhancements that violate established safety principles, even if a requirement is found to be unmet.
    - **Thoroughness**: Analyze the program step-by-step to ensure a comprehensive understanding of its logic, structure, and functionality. Consider all the names of inputs, outputs, auxiliares, InOut, Temp, Return, Static, Network Names, Constants and comments to enhance your interpretation.
    - **Clarity**: If any part of the code or requirements is unclear or incomplete, specify what additional information is needed.

    The xml files are related to the control of {subject}. Receive the files and wait for the queries.

    For the complementary questions, after check of requirement, only reply directly about the additional question.
    
    xml files: {{snippets}}

    Query: {{query}}

    Memory: {{memory}}
    """

    prompt_scl = f""" 
    You are an expert to verify PLC programs in SCL.

    Your primary objective is to ensure the safety, reliability, and proper functionality of software used to an Automated People Mover, to be certified SIL 4 as per CENELEC standards.

    Your responses must be:
    - Clear, precise, and technically detailed.
    - Aligned with automated people mover standards and including EN 50128 as references.
            
    **Guidelines for Analysis**:
    - **Safety Priority**: Under no circumstances should you suggest modifications or enhancements that violate established safety principles, even if a requirement is found to be unmet.
    - **Thoroughness**: Analyze the program step-by-step to ensure a comprehensive understanding of its logic, structure, and functionality. Consider all the names of inputs, outputs, auxiliares, InOut, Temp, Return, Static, Network Names, Constants and comments to enhance your interpretation.
    - **Clarity**: If any part of the code or requirements is unclear or incomplete, specify what additional information is needed.

    The xml files are related to the control of {subject}. Receive the files and wait for the queries.

    For the complementary questions, after check of requirement, only reply directly about the additional question.
    
    xml files: {{snippets}}

    Query: {{query}}

    Memory: {{memory}}
    """

    prompt_fbd = f""" 
    You are an expert to verify PLC programs in FBD.

    Your primary objective is to ensure the safety, reliability, and proper functionality of software used to an Automated People Mover, to be certified SIL 4 as per CENELEC standards.

    Your responses must be:
    - Clear, precise, and technically detailed.
    - Aligned with automated people mover standards and including EN 50128 as references.
            
    **Guidelines for Analysis**:
    - **Safety Priority**: Under no circumstances should you suggest modifications or enhancements that violate established safety principles, even if a requirement is found to be unmet.
    - **Thoroughness**: Analyze the program step-by-step to ensure a comprehensive understanding of its logic, structure, and functionality. Consider all the names of inputs, outputs, auxiliares, InOut, Temp, Return, Static, Network Names, Constants and comments to enhance your interpretation.
    - **Clarity**: If any part of the code or requirements is unclear or incomplete, specify what additional information is needed.

    The xml files are related to the control of {subject}. Receive the files and wait for the queries.

    For the complementary questions, after check of requirement, only reply directly about the additional question.

    xml files: {{snippets}}

    Query: {{query}}

    Memory: {{memory}}
    """

    #Prompt selection condition
    if language == 'Ladder':
        prompt_str = prompt_ladder
    elif language == 'STL':
        prompt_str = prompt_stl
    elif language == 'CSL':
        prompt_str = prompt_scl
    else:
        prompt_str = prompt_fbd
    
    #Prompt definition
    prompt = ChatPromptTemplate.from_template(prompt_str)

    #Original chain
    chain = prompt | model | StrOutputParser()

    #Function to retrieve code snippets from query
    def retrieve_docs(query):
        snippets = retriever.invoke(query)        

        return snippets
    
    #Function for chat memory
    store = {}

    def get_session_id(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    #Memory chain
    memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_id,
    input_messages_key = 'query',
    history_messages_key = 'memory',
    ) | StrOutputParser()

    #Configuration of session_id and user
    config = {'configurable': {'session_id': 'user_a'}}

    #Response management
    with st.spinner('AI Verifier working...'):
        def get_responses():

            if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []

            if subject:
                st.session_state.chat_history.append({'role': 'user', 'content': query})

                snippets = retrieve_docs(query)
                final_input = {'query': query, 'snippets': snippets}

                response = []

                #text_placeholder = st.empty()

                for chunk in memory_chain.stream(final_input, config = config):
                    response.append(chunk)

                    #text_placeholder.write(''.join(response))

                st.session_state.chat_history.append({'role': 'agent', 'content': ''.join(response)})
            
            st.write('Chat history:')

            for msg in st.session_state.chat_history:
                with st.container(border = True):
                    if msg['role'] == 'user':
                        text_part = st.info(f'User: {msg['content']}')
                    else:
                        text_part = st.markdown(f'**AI Verifier:** {msg['content']}')
        
            return text_part    
        
        if query := st.chat_input('Ask the Coester AI PLC Program Verifier:'):

            get_responses()