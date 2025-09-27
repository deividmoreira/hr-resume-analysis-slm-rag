# Projeto 4 - App de IA Generativa com SLM, RAG e Engenharia de Prompt Para Assistente de RH (Análise de Currículos com IA)

# Imports
import docx2txt  
import streamlit as st  
from langchain_huggingface import HuggingFaceEmbeddings  
from llama_index.llms.ollama import Ollama  
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 

# Configuração da página
st.set_page_config(page_title = "Data Science Academy", page_icon = ":100:", layout = "centered")  

# Barra lateral com instruções
st.sidebar.title("Instruções")
st.sidebar.write("""
- Faça upload dos currículos no diretório `./documentos`.
- Digite perguntas específicas sobre os candidatos para obter respostas detalhadas com base na análise dos currículos.
- O assistente utiliza técnicas avançadas de Engenharia de Prompt e RAG para garantir a precisão das respostas.
- O processamento inicial dos documentos pode levar entre 1 e 2 minutos. Seja paciente e aguarde.
""")

# Botão de suporte na barra lateral
if st.sidebar.button("Suporte"):
    st.sidebar.write("Dúvidas? Envie um e-mail para: suporte@datascienceacademy.com.br")

# Título principal da aplicação
st.title("DSA - Projeto 4")  
st.title("App de IA Generativa com SLM, RAG e Engenharia de Prompt Para Assistente de RH (Análise de Currículos com IA)") 

# Verifica se a chave "messages" não está presente no session_state do Streamlit
if "messages" not in st.session_state.keys():  

    # Inicializa a chave "messages" com uma mensagem padrão do assistente
    st.session_state.messages = [
        {"role": "assistant", "content": "Digite sua pergunta"}  
    ]

# Inicializa um modelo de linguagem da Ollama chamado "gemma3:4b" com um tempo limite de 600 segundos
llm = Ollama(model="gemma3:4b", request_timeout=600.0)  

# Define o modelo de embeddings da Hugging Face para transformação de sentenças
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  
 
# Usa o decorador @st.cache_resource para armazenar em cache o resultado da função e evitar recarregamentos desnecessários
@st.cache_resource(show_spinner=False)
def dsa_modulo_rag():
    
    # Exibe um spinner com uma mensagem enquanto os documentos são carregados e indexados
    with st.spinner(text="Carregando e indexando os documentos Streamlit – aguarde! Isso deve levar de 1 a 2 minutos."):  
        
        # Lê os documentos do diretório especificado, permitindo busca recursiva em subpastas
        reader = SimpleDirectoryReader(input_dir="./documentos", recursive=True)  
        
        # Carrega os dados dos documentos
        docs = reader.load_data()  
        
        # Define o modelo de linguagem globalmente nas configurações
        Settings.llm = llm  
        
        # Define o modelo de embeddings globalmente nas configurações
        Settings.embed_model = embed_model  
        
        # Cria um índice vetorial a partir dos documentos carregados
        index = VectorStoreIndex.from_documents(docs)  
        
        # Retorna o índice criado
        return index  

# Executa a função
index = dsa_modulo_rag()  

# Verifica se a chave "chat_engine" não está presente no session_state do Streamlit
if "chat_engine" not in st.session_state.keys():  

    # Inicializa o chat engine a partir do índice vetorial com modo de condensação de perguntas e saída detalhada
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)  

# Se a chave "chat_engine" não for encontrada, o código cria e armazena um chat engine baseado em um índice vetorial. 
# Esse chat engine é configurado para operar no modo "condense_question", o que significa que ele reformula as perguntas do usuário 
# para garantir um melhor contexto antes de gerar uma resposta. Além disso, o parâmetro verbose=True é ativado, permitindo que o modelo 
# forneça saídas detalhadas e explicativas sobre o processo de geração de respostas. Dessa forma, o chat engine fica pronto para interagir 
# com o usuário e responder de forma mais precisa e eficiente.

# Exibe um campo de entrada para o usuário digitar sua pergunta e armazena a entrada na variável 'prompt'
if prompt := st.chat_input("Sua pergunta"):  

    # Adiciona a pergunta do usuário à lista de mensagens no session_state
    st.session_state.messages.append({"role": "user", "content": prompt})  

# Percorre todas as mensagens armazenadas no session_state
for message in st.session_state.messages: 

    # Exibe cada mensagem no chat, diferenciando entre usuário e assistente
    with st.chat_message(message["role"]):  

        # Escreve o conteúdo da mensagem na interface do chat
        st.write(message["content"])  

# Verifica se a última mensagem armazenada no session_state não foi enviada pelo assistente
if st.session_state.messages[-1]["role"] != "assistant":

    # Cria um bloco de mensagem para o assistente no chat
    with st.chat_message("assistant"):

        # Exibe um indicador de carregamento enquanto a resposta está sendo processada
        with st.spinner("Pensando..."):

            # Obtém o conteúdo da última mensagem enviada pelo usuário
            user_message = st.session_state.messages[-1]["content"]
            
            # Cria um prompt contextualizado para o assistente de RH, considerando currículos disponíveis
            contextual_prompt = f"Você é um assistente de RH especializado em análise de currículos de candidatos. O usuário fez a seguinte pergunta: '{user_message}'. Considere todos os documentos (currículos) disponíveis e forneça uma resposta detalhada e precisa sempre em Português do Brasil."
            
            # Obtém a resposta do chat engine com base no prompt contextualizado
            response = st.session_state.chat_engine.chat(contextual_prompt)
            
            # Exibe a resposta do assistente no chat
            st.write(response.response)
            
            # Armazena a resposta do assistente no session_state para manter o histórico da conversa
            st.session_state.messages.append({"role": "assistant", "content": response.response})  


