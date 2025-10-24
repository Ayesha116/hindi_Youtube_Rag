from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
import streamlit as st
import os
import re
from dotenv import load_dotenv

os.environ["LANGCHAIN_PROJECT"] = "hindi-youtube-rag"

# --- STREAMLIT SETUP ---

st.set_page_config(page_title="YouTube Q&A with RAG", page_icon="▶️")

load_dotenv()


# --- LLM & EMBEDDINGS ---

trans_llm = ChatGroq(model ="llama-3.1-8b-instant" , temperature=1)
response_llm = ChatGroq(model ="groq/compound" , temperature=0.5)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")


# --- STREAMLIT UI ---

st.title("YouTube Video Q&A 🎬")

st.write("Enter a Hindi YouTube video URL and ask a question in Roman Hindi.")

# --- USER INPUT ---

video_url = st.text_input("Enter Hindi YouTube video URL:")
question = st.text_input("Enter your question in Hindi (Roman script):")


pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
match = re.search(pattern, video_url)
if match:
    video_id = match.group(1)
    

# --- PROCESSING & OUTPUT ---

transcript_list = []

if st.button("Get Answer"):
    if video_id and question:
        with st.spinner("Fetching transcript and processing..."):

            try:
                yt_api = YouTubeTranscriptApi()
                fetched_transcript = yt_api.fetch(video_id=video_id, languages=['hi'])

                for segment in fetched_transcript:
                    transcript_list.append(segment.text)

                transcript = ' '.join(transcript_list)

            except TranscriptsDisabled:
                st.error("Transcripts are disabled for this video.")
                st.stop()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )

            text = text_splitter.split_text(transcript)

            vectorstore = FAISS.from_texts(texts=text, embedding=embeddings)

            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

            parser = StrOutputParser()

            translation_prompt = PromptTemplate(
                template="transliterate the following text to devanagri hindi script: {question}. Provide only the transliterated text without any additional commentary.",
                input_variables=["question"]
            )
            main_prompt = PromptTemplate(
                template="You are a helpful assistant that helps people find information. Use the following pieces of context to answer the question. your answer should be in roman hindi as user can't read hindi. If you don't know the answer, just say that you don't know, don't try to make up an answer. \n\ncontext:{context}\n\nQuestion: {question}\"nHelpful answer in markdown:",
                input_variables=["context", "question"]
            )

            def response_cleaner(response: str) -> str:
                cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                return cleaned_response

            def format_retrieved_text(retrieved_chunks)-> str:
                combined_text = " ".join([chunk.page_content for chunk in retrieved_chunks])
                return combined_text
            
            trans_chain = translation_prompt | trans_llm | parser | RunnableLambda(response_cleaner)

            parallel_chain = RunnableParallel({
                'context': retriever |RunnableLambda(format_retrieved_text) ,
                'question': RunnablePassthrough(), 
            })

            main_chain = trans_chain | parallel_chain| main_prompt | response_llm | parser | RunnableLambda(response_cleaner)
            
            result = main_chain.invoke({'question': question})
            
            st.write(result)


