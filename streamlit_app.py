#  streamlit run streamlit_app.py 
# streamlit_app.py
import os
import sys
import streamlit as st
from dotenv import load_dotenv
import json
import datetime
import uuid
import random

# Add project root directory to Python path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import modules
from src.index_graph.graph import build_index
from src.retrieval_graph.graph import build_chain

# Load environment variables from .env file (using correct absolute path)
def load_env():
    """Load environment variables using absolute path to ensure correct loading"""
    # Get project root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(root_dir, '.env')
    
    print(f"Attempting to load environment variables from: {env_path}")
    load_dotenv(env_path)
    
    # Debug information
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        key_preview = f"{openai_api_key[:5]}...{openai_api_key[-4:]}"
        print(f"Loaded OPENAI_API_KEY from {env_path}: {key_preview}")
    else:
        print(f"Warning: Failed to load OPENAI_API_KEY from {env_path}")

# Save chat history locally
def save_chat_history(query, answer, sources, session_id=None):
    """
    Save chat history to local file
    
    Parameters:
        query: User question
        answer: System response
        sources: Source information
        session_id: Session ID, auto-generated if None
    """
    # Ensure logs directory exists
    logs_dir = os.path.join(root_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a session ID if none exists
    if not session_id:
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        session_id = st.session_state.session_id
    
    # Prepare record data
    timestamp = datetime.datetime.now().isoformat()
    chat_record = {
        "timestamp": timestamp,
        "session_id": session_id,
        "query": query,
        "answer": answer,
        "sources": sources
    }
    
    # Save to date-named file
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(logs_dir, f"chat_history_{date_str}.jsonl")
    
    # Write to file in append mode
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(chat_record, ensure_ascii=False) + '\n')
    
    print(f"Chat history saved to: {log_file}")

# Load chat history
def load_chat_history(days_back=7):
    """
    Load recent chat history from log files
    
    Parameters:
        days_back: Number of days to look back for history
    
    Returns:
        List of chat records sorted by timestamp (newest first)
    """
    logs_dir = os.path.join(root_dir, 'logs')
    history_records = []
    
    if not os.path.exists(logs_dir):
        return history_records
    
    # Get recent dates
    for i in range(days_back):
        date = datetime.datetime.now() - datetime.timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        log_file = os.path.join(logs_dir, f"chat_history_{date_str}.jsonl")
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line.strip())
                            history_records.append(record)
            except Exception as e:
                print(f"Error loading history from {log_file}: {e}")
    
    # Sort by timestamp (newest first)
    history_records.sort(key=lambda x: x['timestamp'], reverse=True)
    return history_records

# Call the environment variable loading function
load_env()

# Set Streamlit page configuration
st.set_page_config(
    page_title="CDC Weekly Reports RAG", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.3rem;
        color: #666666;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f7ff;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #0066cc;
    }
    .example-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
        cursor: pointer;
    }
    .example-box:hover {
        background-color: #e9ecef;
        border-color: #adb5bd;
    }
    .source-header {
        font-weight: bold;
        color: #495057;
    }
    .stButton button {
        width: 100%;
    }
    .chat-separator {
        margin: 1rem 0;
        border-top: 1px solid #e9ecef;
    }
    .footer {
        margin-top: 2rem;
        text-align: center;
        color: #6c757d;
        font-size: 0.9rem;
    }
    /* 把侧边栏顶部内边距减到最小 */
    [data-testid="stSidebar"] .block-container {
        padding-top: 0px !important;
    }
    /* 减少侧边栏中各元素的间距 */
    [data-testid="stSidebar"] .block-container > div {
        padding-top: 0px !important;
        gap: 0.5rem !important;
    }
    /* 减少markdown元素的边距 */
    [data-testid="stSidebar"] .block-container p {
        margin-bottom: 0.3rem !important;
        margin-top: 0.3rem !important;
    }
    /* 减少分隔线的间距 */
    [data-testid="stSidebar"] hr {
        margin: 0.5rem 0 !important;
    }
    /* 调整标题间距 */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Create two columns for layout - sidebar and main content
with st.sidebar:
    # 极小化CDC标题上下的空白
    st.markdown("""
    <div style="text-align: center; margin-top: 0; margin-bottom: 0;">
        <h2 style="color: #0066cc; margin-bottom: 2px; margin-top: 0;">CDC</h2>
        <h4 style="color: #555555; margin-top: 0; margin-bottom: 2px;">Weekly Reports System</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='margin-top: 0; font-size: 1.2rem;'>About This System</h2>", unsafe_allow_html=True)
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) system allows you to ask questions about CDC Weekly Reports data.
    
    The system will:
    - Search for relevant information in the CDC document database
    - Provide accurate answers based on retrieved content
    - Show citations to the source documents
    """)
    
    st.markdown("---")
    st.markdown("### Example Questions")
    
    # Example questions
    example_questions = [
         "What did the CDC report about influenza activity among children?",
        "What did the Tennessee report say about early influenza activity in children?",
        
        "Arthritis—how much higher in young male veterans?",
        # "COPD fell in which U.S. age group, and by how much?",
        "Why do the authors prioritize the WHO \"STOP AIDS\" package for children under 5 on ART, and which two gaps drove the excess deaths?",
        # "Which adult age group had a significant decline in COPD prevalence from 2011–2021, and what was the AAPC?"
    ]
    
    
    def set_example_question(question):
        st.session_state.example_question = question
    
    for question in example_questions:
        st.button(question, key=f"btn_{hash(question)}", on_click=set_example_question, args=(question,))
    
    st.markdown("---")
    st.markdown("### System Information")
    st.markdown(f"Session ID: `{st.session_state.get('session_id', 'Not set')}`")
    st.markdown(f"Questions asked: `{len(st.session_state.get('history', []))//2}`")
    
    st.markdown("---")
    st.markdown("### 📜 Recent Chat History")
    
    # Load and display recent chat history
    try:
        recent_history = load_chat_history(days_back=3)  # Show last 3 days
        
        if recent_history:
            # Show total count
            st.markdown(f"**Total recent conversations: {len(recent_history)}**")
            
            # Create expandable sections by date
            history_by_date = {}
            for record in recent_history:
                date_str = record['timestamp'][:10]  # Extract date part (YYYY-MM-DD)
                if date_str not in history_by_date:
                    history_by_date[date_str] = []
                history_by_date[date_str].append(record)
            
            # Display history grouped by date
            for date_str in sorted(history_by_date.keys(), reverse=True):
                day_records = history_by_date[date_str]
                with st.expander(f"📅 {date_str} ({len(day_records)} questions)", expanded=(date_str == datetime.datetime.now().strftime("%Y-%m-%d"))):
                    for i, record in enumerate(day_records[:10]):  # Show max 10 per day
                        # Format timestamp
                        time_str = record['timestamp'][11:19]  # Extract time part (HH:MM:SS)
                        
                        # Truncate long questions
                        question = record['query']
                        if len(question) > 60:
                            question = question[:60] + "..."
                        
                        # Create clickable question
                        if st.button(f"{time_str}: {question}", key=f"hist_{record['timestamp']}_{i}", help=f"Full question: {record['query']}"):
                            # Set the clicked question as the example question
                            st.session_state.example_question = record['query']
                            st.rerun()
                    
                    if len(day_records) > 10:
                        st.markdown(f"*... and {len(day_records) - 10} more questions*")
        else:
            st.markdown("*No recent chat history found*")
            
    except Exception as e:
        st.markdown("*Error loading chat history*")
        print(f"Error in chat history display: {e}")

# Main content area
st.markdown('<h1 class="main-header">📊 CDC Weekly Reports Intelligence System</h1>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Ask questions about CDC reports, emerging health trends, and public health guidance</div>', unsafe_allow_html=True)

# Info box with guidance
if len(st.session_state.get("history", [])) == 0:
    st.markdown("""
    <div class="info-box">
        <h4>Getting Started</h4>
        <p>This system provides answers to your questions about CDC Weekly Reports, drawing directly from official published content.</p>
        <p>Try asking about disease trends, public health recommendations, or emerging health concerns.</p>
        <p>You can select an example question from the sidebar or type your own question below.</p>
    </div>
    """, unsafe_allow_html=True)

# Data loading
try:
    # Only call build_index once
    with st.spinner("Loading CDC data..."):
        index = build_index()
        chain = build_chain()
except Exception as e:
    st.error("Error initializing the system")
    st.exception(e)
    st.stop()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Ensure session ID exists
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    print(f"New session started, ID: {st.session_state.session_id}")

# Chat interface
if "example_question" in st.session_state:
    example = st.session_state.example_question
    query = st.chat_input("Enter your question...")
    
    if not query and "processed_example" not in st.session_state:
        query = example
        st.session_state.processed_example = True
        
    if "processed_example" in st.session_state and st.session_state.processed_example:
        del st.session_state.example_question
        del st.session_state.processed_example
else:
    query = st.chat_input("Enter your question about CDC Weekly Reports...")

# Display chat history
for i, (role, msg) in enumerate(st.session_state.history):
    with st.chat_message(role):
        st.markdown(msg)
    
    # Add separator after each Q&A pair except the last one
    if i % 2 == 1 and i < len(st.session_state.history) - 1:
        st.markdown('<div class="chat-separator"></div>', unsafe_allow_html=True)

# Process new query
if query:
    # Add user message to history
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching CDC reports..."):
            resp = chain.invoke({"query": query, **index})
            answer, sources = resp["answer"], resp["sources"]
            detailed_sources = resp.get("detailed_sources", [])

        st.markdown(answer)
        
        with st.expander("📎 View Sources"):
            st.markdown('<p class="source-header">This answer draws from the following CDC documents:</p>', unsafe_allow_html=True)
            
            unique_sources = []
            seen = set()
            for f, cid in sources:
                source_key = f"{f}_{cid}"  
                if source_key not in seen:
                    seen.add(source_key)
                    unique_sources.append((f, cid))
            
            for f, cid in unique_sources:
                st.markdown(f"- `{f}`  chunk {cid}")

    # Add assistant response to history
    st.session_state.history.append(("assistant", resp["answer"]))
    
    # Save chat history
    save_chat_history(query, resp["answer"], detailed_sources, st.session_state.session_id)

# Footer
st.markdown('<div class="footer">CDC Weekly Reports RAG System | Data updated regularly</div>', unsafe_allow_html=True)
