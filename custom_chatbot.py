import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


model_id="meta-llama/Meta-Llama-3-8B-Instruct"
instructions="If the user asks anything that is not there in the document, it should just tell them to contact the business directly."

def get_text_chunks_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):

    llm = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature":0.3, "max_length":512})
    llm.client.api_url = 'https://api-inference.huggingface.co/models/'+model_id
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': instructions+" "+user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content.replace(instructions,"")), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vs" not in st.session_state:
        st.session_state.vs = None

    st.header("Ask a question from your PDF")
    user_question = st.text_input("Model id "+model_id )

    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):

                text_chunks = get_text_chunks_from_pdf(pdf_docs)
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vectorstore
                st.success('File uploaded, chunked and embedded successfully.')

        st.subheader("Create new Session with embeddings")
        if st.button("New chat"):
                st.session_state.conversation = get_conversation_chain(st.session_state.vs)

if __name__ == '__main__':
    main()
