import streamlit as st
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# RAG Dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "./my_finetuned_model"
PDF_PATH = "rag.pdf"
DB_PATH = "faiss_index"

# Page Setup
st.set_page_config(page_title="Sapt-Sindhu Storyteller", page_icon="📜", layout="wide")
st.title("📜 Sapt-Sindhu Storyteller")

# RAG Setup
@st.cache_resource
def setup_rag():
    status = st.empty()
    status.info("Checking Knowledge Base...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load existing index if available
        if os.path.exists(DB_PATH):
            db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
            status.success("Knowledge Base Loaded!")
        # Create new index if not found
        else:
            status.info("Indexing PDF...")
            if not os.path.exists(PDF_PATH):
                status.error(f"PDF not found: {PDF_PATH}")
                return None
            loader = PyPDFLoader(PDF_PATH)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(DB_PATH)
            status.success("Knowledge Base Indexed & Loaded!")
        
        status.empty()
        return db
    except Exception as e:
        status.error(f"RAG Setup Error: {e}")
        return None

# Model Loading
@st.cache_resource
def load_model():
    status_text = st.empty()
    status_text.info("Loading AI Model...")
    
    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map=device,  
            torch_dtype=torch.float16 if device != "cpu" else torch.float32, 
            trust_remote_code=False,
            attn_implementation="eager" 
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
        status_text.success(f"Model Loaded on {device.upper()}!")
        status_text.empty()
        return model, tokenizer, device
    except Exception as e:
        status_text.error(f"Error loading model: {e}")
        return None, None, device

# Initialize Resources
db = setup_rag()
model, tokenizer, device = load_model()

# Sidebar Controls
with st.sidebar:
    st.header("Story Parameters")
    
    temperature = st.slider("Creativity", 0.1, 1.0, 0.6)
    max_tokens = st.slider("Length", 256, 1500, 800)
    
    st.divider()
    st.header("Context Selectors")
    
    # Dropdowns for context constraints
    theme_options = [
        "Any (Auto-detect)", "Waahadat-ul-Wajood (Unity of Being)", 
        "Anekantavada (Many-Sided Truth)", "Prem (Transformative Love)", 
        "Wand Chako (Share and Uplift)", "Detached Worldliness", 
        "Saint-Warrior Ideal", "Teacher-Disciple Relationship"
    ]
    selected_theme = st.selectbox("Select Moral Theme:", theme_options)
    
    figure_options = [
        "Any (Auto-detect)", "Guru Nanak", "Mirabai", "Bhai Mardana", 
        "Dara Shikoh", "Bulleh Shah", "Baba Banda Singh Bahadur", 
        "Sai Miya Mir", "Shah Abdul Latif Bhittai"
    ]
    selected_figure = st.selectbox("Select Historical Figure:", figure_options)
    
    st.divider()
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main Logic
if user_input := st.chat_input("Request a story (e.g., 'A story about courage'):"):
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Construct constraints for RAG
    constraints = []
    if selected_theme != "Any (Auto-detect)":
        constraints.append(f"Focus on the theme: {selected_theme}")
    if selected_figure != "Any (Auto-detect)":
        constraints.append(f"Include the figure: {selected_figure}")
        
    augmented_prompt = user_input
    if constraints:
        constraint_str = ". ".join(constraints)
        augmented_prompt = f"{user_input}. (Context instructions: {constraint_str})"

    # Generation process
    if model and tokenizer and db:
        with st.chat_message("assistant"):
            container = st.container()
            with container:
                status_box = st.status("Thinking...", expanded=True)
                
                try:
                    # Retrieve relevant context
                    status_box.write("Searching knowledge base...")
                    retrieved_docs = db.similarity_search(augmented_prompt, k=4)
                    context_str = "\n".join([f"[Source Chunk]: {d.page_content}" for d in retrieved_docs])
                    
                    # Prepare prompts
                    status_box.write("Drafting story...")
                    system_prompt = (
                        "You are a Sapt-Sindhu civilizational storytelling model.\n"
                        "You do not invent morals. You only reason using the retrieved moral–historical canon provided below.\n"
                        "Strictly follow this structure:\n"
                        "dont use historical figures directly as characters.\n"
                        "STORY\n[Write the story here]\n"
                        "MEANING\n[Explain the morals here]\n"
                        "Do NOT write a 'Retrieved Moral Canon' section yourself."
                    )
                    
                    user_prompt = f"""Use the following Retrieved Canon to answer the User Request.

Retrieved Canon:
{context_str}

User Request: {user_input}
Additional Constraints: {constraints}
"""
                    
                    chat_struct = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    # Tokenize and Generate
                    input_text = tokenizer.apply_chat_template(chat_struct, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(input_text, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=temperature,
                            use_cache=True
                        )
                    
                    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    
                    # Post-processing and display
                    status_box.write("Finalizing...")
                    if "MEANING" in generated_text:
                        parts = generated_text.split("MEANING")
                        story_text = parts[0].replace("STORY", "").strip()
                        meaning_text = parts[1].strip()
                    else:
                        story_text = generated_text.replace("STORY", "").strip()
                        meaning_text = "Meaning not explicitly separated."
                    
                    status_box.update(label="Complete!", state="complete", expanded=False)
                    
                    st.subheader("STORY")
                    st.write(story_text)
                    st.divider()
                    st.subheader("MEANING")
                    st.write(meaning_text)
                    
                    full_log = f"**STORY**\n{story_text}\n\n**MEANING**\n{meaning_text}"
                    st.session_state.messages.append({"role": "assistant", "content": full_log})
                    
                except Exception as e:
                    status_box.update(label="Error", state="error")
                    st.error(f"Generation Error: {e}")
    else:
        st.error("System not fully initialized.")