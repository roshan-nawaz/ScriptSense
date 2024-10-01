import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


from files.htr_model import load_and_predict_images, create_pdf
from files.line_splitting import line_split
from files.css import CSS


# Load environment variables from .env file
load_dotenv()
# os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Inject CSS
st.markdown(CSS, unsafe_allow_html=True)

# Function to handle vector embedding
def chatbot():
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader(OUTPUT_FOLDER)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:3])
    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
    st.session_state.embedding_done = True
    st.write("Chatbot is Ready to Answer your Questions.")

# Initialize session state variables if not already initialized
if 'upload_button' not in st.session_state:
    st.session_state.upload_button = False

if 'embedding_done' not in st.session_state:
    st.session_state.embedding_done = False

if 'exit_clicked' not in st.session_state:
    st.session_state.exit_clicked = False

if 'recognize_text' not in st.session_state:
    st.session_state.recognize_text = ""

def click_upload_button():
    st.session_state.upload_button = True

# File upload handler
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Function for speech recognition
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        with st.spinner("Listening..."):
            audio = r.listen(source)
        with st.spinner("Recognizing..."):
            try:
                text = r.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                st.error("Google Speech Recognition could not understand the audio")
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service; {e}")
    return ""


# Main app logic
def main():
    if st.session_state.exit_clicked:
        st.session_state.clear()
        st.title("Thank you and Have a Nice Day.....")
        st.balloons()
        st.stop()

    st.markdown('<h1 id="htr-title">Handwritten Text Recognition (HTR)</h1>', unsafe_allow_html=True)

    # Show file uploader if file has not been uploaded yet
    if not st.session_state.upload_button:
        st.header("Upload a PDF/IMG file")
        uploaded_file = st.file_uploader("Choose a PDF/IMG file", type=["pdf", "png", "jpg", "jpeg"])

        if uploaded_file is not None:
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File uploaded successfully.")
            st.session_state.uploaded_file_path = file_path
            output_pdf_path = os.path.join(OUTPUT_FOLDER, 'digital_text.pdf')
            with st.spinner('Converting...'):

                line_imgs_path = line_split(file_path)
                predicted_text = load_and_predict_images(line_imgs_path)
                create_pdf(predicted_text, output_pdf_path)
            click_upload_button()
            st.success("Successfully file converted to pdf.")
            with open(output_pdf_path, "rb") as file:
                st.download_button(label="Download Converted PDF", data=file, file_name='digital_text.pdf', mime='application/pdf')


    # Show "Documents Embedding" button if embedding is not done yet
    if st.session_state.upload_button and not st.session_state.embedding_done:
        if st.button("Chatbot"):
            with st.spinner('Loading...'):
                chatbot()
            

    # Show the new prompt input and exit button after embedding is done
    if st.session_state.embedding_done:
        with st.form(key='speech_form'):
            if st.form_submit_button("Use Speech Recognition"):
                st.session_state.recognize_text = recognize_speech()
                st.rerun()

        new_prompt = st.text_input("Enter Your Question From Documents", st.session_state.recognize_text)

        # Handle user input and retrieve answers 
        if new_prompt:
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
            prompt = ChatPromptTemplate.from_template(
                """
                analyze the input text with spelling mistakes. Correct the spelling errors and provide answer for context based on input only do not display the original, analyzed and corrected in the output.
                {input}
                {context}

                """
            )

            
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({'input': new_prompt})
            output_text = response["answer"]
            
            with st.spinner('Running...'):
                st.write(output_text)

                # Display references from the document
                with st.expander("References from Document:"):
                    for doc in response["context"]:
                        st.write(doc.page_content)
                        st.write("--------------------------------")
        
        # Add exit button at the bottom
        if st.button("Exit"):
            st.session_state.exit_clicked = True
            st.rerun()

if __name__ == "__main__":
    main()
