import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
import time
import datetime
import random

# Page configuration
st.set_page_config(
    page_title="Intelligent Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Claude-like styling with modern UI elements
st.markdown("""
<style>
    /* Modern clean interface */
    html, body, .main {
        background-color: #f9f9f9;
        color: #333333;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header styling */
    h1, h2, h3 {
        font-weight: 600;
        color: #333;
    }

    /* Chat container */
    .stChatMessage {
        max-width: 800px;
        margin: 0 auto 12px auto;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        word-wrap: break-word;
        white-space: pre-wrap;
        line-height: 1.5;
    }

    /* User message styling */
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #e6f7ff;
        border-left: 3px solid #0096CF;
    }

    /* AI message styling */
    .stChatMessage[data-testid="chat-message-ai"] {
        background-color: #ffffff;
        border-left: 3px solid #7f58af;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #ffffff;
        color: #333333;
        padding: 20px;
    }

    /* Input elements */
    .stSelectbox div[data-baseweb="select"], .stTextInput textarea {
        background-color: #ffffff !important;
        border: 1px solid #e1e4e8 !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }

    /* Chat input container */
    .stChatInputContainer {
        max-width: 800px;
        margin: 0 auto;
        background-color: #ffffff;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        border: 1px solid #e1e4e8;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }

    /* Button styling */
    button[data-testid="baseButton-secondary"] {
        background-color: #7f58af;
        color: white;
        border-radius: 8px;
        border: none;
        transition: all 0.2s ease;
    }
    
    button[data-testid="baseButton-secondary"]:hover {
        background-color: #6a4a9e;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer {
        visibility: hidden;
    }
    
    /* Custom progress bar for thinking animation */
    .thinking-animation {
        width: 100%;
        height: 3px;
        background: linear-gradient(to right, #7f58af, #c969a6);
        background-size: 200% 100%;
        animation: thinking-gradient 2s linear infinite;
        border-radius: 3px;
        margin: 10px 0;
    }
    
    @keyframes thinking-gradient {
        0% {background-position: 100% 0;}
        100% {background-position: -100% 0;}
    }
    
    /* Message divider */
    .message-divider {
        max-width: 800px;
        margin: 0 auto;
        border-bottom: 1px solid #f0f0f0;
        margin-bottom: 15px;
    }
    
    /* Message timestamp */
    .message-timestamp {
        font-size: 0.75rem;
        color: #888;
        text-align: right;
        margin-top: 5px;
    }
    
    /* Code block styling */
    pre {
        background-color: #f6f8fa !important;
        border-radius: 6px !important;
        padding: 12px !important;
        border: 1px solid #e1e4e8 !important;
        overflow-x: auto !important;
    }
    
    code {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace !important;
        font-size: 0.85em !important;
    }
    
    /* Feedback buttons container */
    .feedback-container {
        display: flex;
        justify-content: flex-end;
        margin-top: 8px;
        gap: 8px;
    }
    
    /* Feedback buttons */
    .feedback-button {
        background-color: transparent;
        border: 1px solid #e1e4e8;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        color: #666;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .feedback-button:hover {
        background-color: #f5f5f5;
        border-color: #ddd;
    }
</style>
""", unsafe_allow_html=True)

# App header
with st.container():
    cols = st.columns([1, 3, 1])
    with cols[1]:
        st.markdown("""
            <div style="text-align: center; padding: 20px 0;">
                <h1 style="margin: 0; font-size: 2.2rem;">Intelligent Assistant</h1>
                <p style="color: #666; margin-top: 5px;">Your professional AI companion for thoughtful conversations</p>
            </div>
        """, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### Configuration")
    
    # Model selection with nice UI
    st.markdown("##### Model Selection")
    selected_model = st.selectbox(
        "Choose your assistant model:",
        ["gemma3:1b", "gemma3:9b", "llama3:8b", "mistral:7b"],
        index=0,
        format_func=lambda x: {
            "gemma3:1b": "Gemma 3 (1B) - Fast responses",
            "gemma3:9b": "Gemma 3 (9B) - Balanced assistant",
            "llama3:8b": "Llama 3 (8B) - Creative assistant",
            "mistral:7b": "Mistral (7B) - Analytical assistant"
        }[x]
    )
    
    # Temperature control
    st.markdown("##### Response Style")
    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower for more deterministic responses, higher for more creative ones"
    )
    
    # System prompt customization
    st.markdown("##### Assistant Personality")
    system_prompt_options = {
        "default": "You are a helpful, harmless, and honest AI assistant. You provide accurate, thoughtful responses.",
        "expert": "You are an expert AI assistant with deep knowledge across multiple domains. Provide detailed, nuanced answers.",
        "creative": "You are a creative AI assistant who thinks outside the box. Offer imaginative perspectives and solutions.",
        "concise": "You are a concise AI assistant. Provide brief, clear answers without unnecessary elaboration."
    }
    
    system_prompt_selection = st.selectbox(
        "Assistant personality:",
        list(system_prompt_options.keys()),
        format_func=lambda x: x.capitalize()
    )
    
    custom_prompt = st.checkbox("Use custom system prompt")
    
    if custom_prompt:
        system_prompt = st.text_area(
            "Custom system prompt:",
            height=100,
            value=system_prompt_options[system_prompt_selection]
        )
    else:
        system_prompt = system_prompt_options[system_prompt_selection]
    
    st.markdown("---")
    
    # Display feature information
    st.markdown("### Features")
    with st.expander("🧠 Capabilities", expanded=False):
        st.markdown("""
        - Thoughtful conversation
        - Code generation and explanation
        - Knowledge questions
        - Writing assistance
        - Problem-solving help
        - Data analysis insights
        """)
    
    # Session management
    st.markdown("### Session")
    if st.button("Clear conversation", type="secondary"):
        st.session_state.message_log = []
        st.session_state.timestamps = []
        st.rerun()

# LLM engine setup
@st.cache_resource
def get_llm_engine(model_name, temp):
    return ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=temp
    )

llm_engine = get_llm_engine(selected_model, temperature)

# Function to format code blocks in messages
def format_code_blocks(text):
    import re
    # Find code blocks - text between triple backticks
    pattern = r"```(.*?)```"
    # Use re.DOTALL to match across multiple lines
    matches = re.findall(pattern, text, re.DOTALL)
    
    formatted_text = text
    for match in matches:
        # Extract language if specified (first word after opening ```)
        parts = match.split("\n", 1)
        if len(parts) > 1:
            lang = parts[0].strip()
            code = parts[1]
        else:
            lang = ""
            code = match
            
        # Replace the original code block with properly formatted markdown
        original = f"```{match}```"
        replacement = f"```{lang}\n{code}\n```"
        formatted_text = formatted_text.replace(original, replacement)
        
    return formatted_text

# Build prompt chain
def build_prompt_chain():
    prompt_sequence = [SystemMessagePromptTemplate.from_template(system_prompt)]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# Generate AI response
def generate_ai_response(prompt_chain):
    start_time = time.time()
    response = (prompt_chain | llm_engine | StrOutputParser()).invoke({})
    response_time = time.time() - start_time
    return response, response_time

# Initialize session state
if "message_log" not in st.session_state:
    st.session_state.message_log = []
    
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

# Display welcome message if conversation is empty
if not st.session_state.message_log:
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    st.session_state.message_log.append({
        "role": "ai", 
        "content": f"Hello! I'm your Intelligent Assistant, ready to help with questions, tasks, or conversations. What can I assist you with today?"
    })
    st.session_state.timestamps.append(current_time)

# Display chat messages
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.message_log):
        with st.chat_message(message["role"]):
            # Format and display message content
            formatted_content = format_code_blocks(message["content"])
            st.markdown(formatted_content)
            
            # Add timestamp if available
            if i < len(st.session_state.timestamps):
                st.markdown(f"<div class='message-timestamp'>{st.session_state.timestamps[i]}</div>", unsafe_allow_html=True)
            
            # Add feedback buttons for AI messages
            if message["role"] == "ai" and i > 0:  # Skip for welcome message
                st.markdown("""
                <div class='feedback-container'>
                    <button class='feedback-button'>👍</button>
                    <button class='feedback-button'>👎</button>
                    <button class='feedback-button'>Copy</button>
                </div>
                """, unsafe_allow_html=True)
        
        # Add divider between messages
        if i < len(st.session_state.message_log) - 1:
            st.markdown("<div class='message-divider'></div>", unsafe_allow_html=True)

# Process user input
user_query = st.chat_input("Ask me anything...")

if user_query:
    # Add user message
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    st.session_state.message_log.append({"role": "user", "content": user_query})
    st.session_state.timestamps.append(current_time)
    
    # Show thinking animation and generate response
    with st.chat_message("ai"):
        with st.container():
            st.markdown("<div class='thinking-animation'></div>", unsafe_allow_html=True)
            thinking_placeholder = st.empty()
            
            # Display thinking messages for a more engaging experience
            thinking_messages = [
                "Analyzing your request...",
                "Processing information...",
                "Considering relevant context...",
                "Formulating response..."
            ]
            
            # Show a sequence of thinking messages
            for msg in thinking_messages:
                thinking_placeholder.markdown(f"*{msg}*")
                time.sleep(0.7)
            
            # Generate the actual response
            prompt_chain = build_prompt_chain()
            ai_response, response_time = generate_ai_response(prompt_chain)
            
            # Clear the thinking animation
            thinking_placeholder.empty()
    
    # Add AI response and timestamp
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.session_state.timestamps.append(current_time)
    
    st.rerun()

# Add information footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; color: #888; font-size: 0.8rem;">
    Powered by Ollama + LangChain • Running locally • Your data stays private
</div>
""", unsafe_allow_html=True)