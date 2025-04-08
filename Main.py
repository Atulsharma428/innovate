import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
import base64
import pandas as pd
import plotly.express as px
import pdfplumber
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.base.llms.types import LLMMetadata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from streamlit_option_menu import option_menu
import requests
import json

# Apply theme from config.toml
st.set_page_config(page_title="GENiE", page_icon="✨", layout="wide")
LITELLM_API_KEY="sk-"
LITELLM_BASE="https://7sqhttk8um.us-east-1.awsapprunner.com"
LLM_MODEL = "gpt-4-m"


# Load CSS file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load CSS file
load_css("styles.css")

def process_text(prompt, input_text):
    
    llm = OpenAI(
        model=LLM_MODEL,
        api_key=LITELLM_API_KEY,
        api_base=LITELLM_BASE,
        max_tokens=4096,
        temperature="1",
    )
    llm.__class__.metadata = LLMMetadata(  # type: ignore[assignment,method-assign]
        context_window=4000,
        num_output=1000,
        is_chat_model=True,
        is_function_calling_model=True,
        model_name=LLM_MODEL,
    )
 
    # Combine the prompt and input_text into a single message
    combined_text = f"{input_text}\n\n{prompt}"
    # Create a chat message
    message = ChatMessage(role="user", content=combined_text)
    # Generate a response using the chat method
    response = llm.chat([message])
    
    return response.message.content





selection = option_menu(
    menu_title=None,  # required
    options=[
        "Home", 
        "Text and Prompt based GenAI App", 
        "Prompt based GenAI app with txt file",
        "Prompt based GenAI app with PDF file", 
        "Prompt based GenAI app with Large and multiple documents"
        ""
    ],  # required
    # icons=["house", "book", "envelope"],  # optional
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
    styles={
        "container": {
            # "padding": "0!important", 
            "background-color": "#ffffff", 
            "display": "flex", 
            "justify-content": "center",
            "width": "100%"
        },
        "icon": {"color": "green", "font-size": "15px", "font-color": "#11b53d"},
        "nav-link": {
            "font-size": "15px",
            "color": "black",  # Set text color to black
            "background-color": "white",  # Set background color to white
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {
            "background-color": "#33c4ff",
            "color": "black"  # Set selected text color to black
        },
    }
)

# Define the pages
# pages = {
#     "Home": "Home",
#     "Single prompt GenAI App": "Single prompt GenAI App",
#     "Single prompt GenAI app with txt file upload": "Single prompt GenAI app with txt file upload",
#     "Single prompt GenAI app with PDF file upload": "Single prompt GenAI app with PDF file upload",
#     "Single prompt GenAI app with Large PDF file upload": "Single prompt GenAI app with Large PDF file upload"
# }

# Function to load Lottie animation JSON file from local storage
def load_lottie_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:  # Specify encoding as utf-8
            lottie_json = json.load(f)
        return lottie_json
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        st.error(f"Invalid JSON format in file: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Path to your Lottie animation JSON file
file_path = "ani.json"  # Replace with your actual file path

# Load the Lottie animation JSON
lottie_json = load_lottie_file(file_path)
# Define the content for the Home page with different font sizes
if selection == "Home":
    # with st.container():
    # st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown('<p class="big-font">Hello! I\'m GENieForge 🖐</p>', unsafe_allow_html=True)
        st.markdown('<p class="small-font">I\'m your friendly AI companion, here to open the gateway to a world of Generative AI RAG (Retrieval-Augmented Generation) applications, effortlessly and delightfully! Imagine a world where complex tasks are as simple as a chat with an old friend. Ready to dive in? Here\'s a sneak peek at the magic we can create together 🚀🚀</p>', unsafe_allow_html=True)

    with right_column:
        if lottie_json:
            st.lottie(lottie_json, speed=1, width=400, height=400)
        else:
            st.warning("No animation loaded.")    
    
    st.markdown('<p class="medium-font">🔍 Summarization Sorcery:</p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Got mountains of text? Watch as I condense it into clear, concise summaries, capturing the essence of your content in the blink of an eye.</p>', unsafe_allow_html=True)

    st.markdown('<p class="medium-font">🔄 Paraphrasing Wizardry:</p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Need a fresh spin on your text? I can rephrase it with style and precision, giving your words a new life while preserving the original meaning.</p>', unsafe_allow_html=True)

    st.markdown('<p class="medium-font">🌐 Translation Marvels:</p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Language barriers? Not anymore! I\'ll translate your text into multiple languages, bridging gaps and connecting worlds with seamless fluency.</p>', unsafe_allow_html=True)

    st.markdown('<p class="medium-font">📄 PDF Chat Magic:</p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Ever wished to have a conversation with your documents? Now you can! Just upload your PDF, and let\'s chat about its contents, extracting valuable information in real-time.</p>', unsafe_allow_html=True)

    st.markdown('<p class="medium-font">🧠 Text Analysis Alchemy:</p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Dive deep into the heart of your text with insightful analysis. Whether it\'s sentiment, keywords, or patterns, I\'ll uncover the hidden gems within.</p>', unsafe_allow_html=True)

    st.markdown('<p class="medium-font">✨ And So Much More:</p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">The possibilities are boundless. Whether you want to create engaging content, develop innovative solutions, or simply explore the wonders of AI, I\'m here to make it all possible.</p>', unsafe_allow_html=True)

    st.markdown('<p class="big-font">Let’s embark on this AI adventure together!</p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">With me by your side, creating Generative AI RAG applications is as easy as a wave of a wand. Ready to transform the way you interact with information? Let\'s get started!</p>', unsafe_allow_html=True)


# **************************************************************************************************************************************************

elif selection == "Text and Prompt based GenAI App":
    st.title("Text and Prompt based GenAI App")

    # Get user inputs
    input_text = st.text_area("Enter input text", height=150)
    prompt = st.text_area("Enter your prompt", height=100)

    if st.button("Get start"):
        if prompt and input_text:
            # Run the process_text function
            result = process_text(prompt, input_text)
            st.subheader("Output")
            st.write(result)
        else:
            st.error("Please enter both prompt and input text.")
# ****************************************************************************************************************************************************

elif selection == "Prompt based GenAI app with txt file":
    st.title("Prompt based GenAI app with txt file upload")
    # Create a file uploader widget
    uploaded_file = st.file_uploader("Upload a Text file")

    # If a file is uploaded, read its contents
    if uploaded_file is not None:
        # Read the file contents as a string
        file_contents = uploaded_file.read().decode("utf-8")
        # Display the file contents
        #st.write(file_contents)
    prompt = st.text_area("Enter your prompt", height=100)

    if st.button("Get start"):
        if prompt and file_contents:
            # Run the process_text function
            result = process_text(prompt, file_contents)
            st.subheader("Output")
            st.write(result)
        else:
            st.error("Please enter both prompt and input text.")


# ****************************************************************************************************************************************************

elif selection == "Prompt based GenAI app with PDF file":
    st.title("Prompt based GenAI app with PDF file upload")
    def extract_data(feed):
        data = []
        try:
            with pdfplumber.open(feed) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            # Convert each table to a DataFrame and append to the data list
                            df = pd.DataFrame(table[1:], columns=table[0])
                            data.append(df)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None 

        if data:
            return pd.concat(data, ignore_index=True)
        else:
            return pd.DataFrame()  
        
    # Streamlit UI
    # Get user inputs
    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    if uploaded_file is not None:
        # Convert the uploaded file to a binary stream
        with pdfplumber.open(uploaded_file) as pdf:
            df = extract_data(uploaded_file)    
            # if df is not None and not df.empty:
            #    st.dataframe(df)  # Display the extracted data 

    prompt = st.text_area("Enter your prompt", height=100)

    if st.button("Get start"):
        if prompt and uploaded_file:
            extracted_text = df.to_string(index=False)  
            result = process_text(prompt, extracted_text)
            st.subheader("Output")
            st.write(result)
        else:
            st.error("Please enter both prompt and input text.")






# ****************************************************************************************************************************************************



elif selection == "Prompt based GenAI app with Large and multiple documents":
    st.title("Prompt based GenAI app with Large and multiple documents file upload")
    def extract_text_and_chunks(feed, chunk_size=500):
        text_chunks = []
        try:
            with pdfplumber.open(feed) as pdf:
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text()
            
            # Split the text into chunks
            for i in range(0, len(full_text), chunk_size):
                text_chunks.append(full_text[i:i+chunk_size])
        except Exception as e:
            st.error(f"An error occurred while extracting text: {e}")
            return None

        return text_chunks

    # Function to generate embeddings for text chunks
    def generate_embeddings(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks, convert_to_tensor=True)
        return embeddings

    # Function to store embeddings in an in-memory vector store
    def store_embeddings_in_memory(chunks, embeddings):
        vector_store = {'chunks': chunks, 'embeddings': embeddings}
        return vector_store


    # Streamlit interface
    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    if uploaded_file is not None:
        # Extract text and create chunks
        st.write("Please wait for few seconds .....")
        text_chunks = extract_text_and_chunks(uploaded_file)
        if text_chunks:
            # Generate embeddings for the text chunks
            embeddings = generate_embeddings(text_chunks)
            
            # Store embeddings in an in-memory vector store
            vector_store = store_embeddings_in_memory(text_chunks, embeddings)
            
            st.write("Text chunks and embeddings have been successfully generated and stored in-memory.")
            
            # Combine text chunks into a single string for processing
            combined_text = " ".join(text_chunks)
            
            # Input prompt from user
            prompt = st.text_area("Enter your prompt", height=100)

            if st.button("Get start"):
                if prompt:
                    result = process_text(prompt, combined_text)
                    st.subheader("Output")
                    st.write(result)
                else:
                    st.error("Please enter both prompt and input text.")

# ********************************************************************************************************************