# type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS 
from langchain_openai.chat_models import ChatOpenAI
from unstructured.partition.auto import partition
from langchain import hub
from dotenv import load_dotenv
import streamlit as st
import logging
import os
load_dotenv()



prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(temperature=0.6, model="gpt-4o-mini")


# Load and extract text from one or multiple PDF/docx/pptx/txt files.
def load_documents(file_paths):
    all_text = []
    for file in file_paths:
        elements = partition(filename=file)
        text_elements = [element.text for element in elements]
        all_text.append("\n\n".join(text_elements))
        
    print(all_text)
    return "\n\n".join(all_text)


# Split a long text into smaller chunks, uses token-based splitting.
def split_text(text: str):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=300,
    )
    print(text_splitter)
    return text_splitter.split_text(text)


# create embeddings and load chunks to vector stores
def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


# Format retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Build and run a Retrieval-Augmented Generation (RAG) chain.
def rag_chain(vectorstore, question):
    qa_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain.invoke(question)


# Generate temporary file path of uploaded docs
def _get_file_path(file_upload):

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the directory exists

    if isinstance(file_upload, str):
        file_path = file_upload  
    else:
        file_path = os.path.join(temp_dir, file_upload.name)
        with open(file_path, "wb") as f:
            f.write(file_upload.getbuffer())
        return file_path


# Main Streamlit app function
def main():
    st.title("Chat with Multiple Documents(pdf, docx, ppt, txt)")
    logging.info("App started")

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi there! How can I help you today?")
            }
        ]


 
    file_upload = st.sidebar.file_uploader(
    label="Upload", type=["pdf", "docx", "pptx","txt"], 
    accept_multiple_files=True,
    key="pdf_uploader"
    )

    if file_upload:     
        st.success("File uploaded successfully! You can now ask your question.")



    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    user_prompt = st.chat_input("Your question")

    # For user message
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Stream assistant response
        with st.chat_message("assistant"):
            logging.info("Generating response...")
            with st.spinner("Processing..."): 
                
 
                file_paths = [_get_file_path(f) for f in file_upload]
                text = load_documents(file_paths)
                chunked_text = split_text(text)
                vectorstores = get_vectorstore(chunked_text)
                assistant_reply = rag_chain(vectorstores, user_prompt)

                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                st.markdown(assistant_reply)




if __name__ == '__main__':
    main()
