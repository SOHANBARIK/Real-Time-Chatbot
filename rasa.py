import streamlit as st
import os
from dotenv import load_dotenv

# --- 1. SETUP & CONFIGURATION (Load Env Vars FIRST) ---
# This must happen before importing LangChain components for Tracing to work correctly.
load_dotenv()

# Optional: verify tracing is active
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print("✅ LangSmith Tracing is ENABLED")

import requests
import uuid
from datetime import datetime
from typing import Annotated, TypedDict

# --- LangGraph & LangChain Imports ---
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Set up the page configuration
st.set_page_config(page_title="KrishnaAI Chatbot", layout="centered")

# --- 2. DEFINE TOOLS & GRAPH (Cached for Performance) ---

@tool
def get_current_time():
    """Returns the current local time."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_weather(city: str):
    """
    Get the current weather for a specific city using Open-Meteo API.
    Args:
        city: The name of the city (e.g., "London", "Dum Dum").
    """
    try:
        # 1. Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_response = requests.get(geo_url).json()
        
        if not geo_response.get("results"):
            return f"Could not find coordinates for {city}."
            
        location = geo_response["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        
        # 2. Get Weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m"
        weather_response = requests.get(weather_url).json()
        
        current = weather_response.get("current", {})
        temp = current.get("temperature_2m", "N/A")
        wind = current.get("wind_speed_10m", "N/A")
        
        return f"Current weather in {city} (Lat: {lat}, Lon: {lon}): {temp}°C, Wind Speed: {wind} km/h."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

@st.cache_resource
def get_graph():
    """
    Initializes the LangGraph.
    st.cache_resource ensures we don't rebuild the graph on every rerun,
    which is crucial for performance and connection stability.
    """
    # Check Critical API Keys
    if not os.getenv("GROQ_API_KEY"):
        st.error("🚨 GROQ_API_KEY is missing in .env")
        st.stop()
    if not os.getenv("TAVILY_API_KEY"):
        st.error("🚨 TAVILY_API_KEY is missing in .env")
        st.stop()
        
    # Check LangSmith Key (Optional warning)
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("⚠️ LANGCHAIN_API_KEY not found. Tracing will not be recorded.")

    # 1. Define Tools
    tools = [TavilySearchResults(max_results=2), get_current_time, get_weather]

    # 2. Initialize LLM (Groq)
    # Tracing happens automatically here because of the env vars
    llm = init_chat_model('llama-3.3-70b-versatile', model_provider='groq', temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # 3. Define State
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # 4. Define Nodes
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # 5. Build Graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools=tools))

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")

    # 6. Compile with Memory
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph

# Load the graph
graph = get_graph()

# --- 3. SESSION STATE MANAGEMENT ---

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Unique Thread ID for this user session (Logs will be grouped by this ID in LangSmith)
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- 4. UI STYLING ---
st.markdown(
    """
    <style>
        /* Dark Theme & Layout */
        body { background-color:rgb(1, 1, 1); color: white; font-family: 'Arial', sans-serif; }
        .stApp { background-color:rgb(1, 1, 1); }
        
        /* Fixed Headers */
        .fixed-top-left {
            position: fixed; top: 10px; left: 10px; font-size: 14px; font-weight: bold;
            color: white; background-color:rgb(0, 0, 0); padding: 5px 10px;
            border-radius: 5px; z-index: 1000;
        }
        .fixed-bottom {
            position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%);
            font-size: 12px; color: white; background-color:rgb(0, 0, 0);
            padding: 5px 10px; border-radius: 5px; z-index: 1000;
        }

        /* Chat Bubbles */
        .message-box { padding: 15px; border-radius: 10px; margin: 10px 0; }
        .user-message { background-color:rgb(20, 20, 20); text-align: right; border: 1px solid #333; }
        .ai-message { background-color: #333; text-align: left; }
        
        /* Input Styling */
        .stTextInput input { background-color: #222; color: white; border: 1px solid #444; }
    </style>

    <div class="fixed-top-left">Krishna AI (LangSmith Tracing Active)</div>
    <div class="fixed-bottom">Made with ❤️ from Sohan</div>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>What can I help with?</h1>", unsafe_allow_html=True)

# --- 5. CHAT LOGIC ---

# Render Chat History
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "ai-message"
        st.markdown(f"<div class='message-box {role_class}'>{message['content']}</div>", unsafe_allow_html=True)

# Input Handling
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("", placeholder="Ask me anything...", key="chat_input_form", label_visibility="collapsed")
    submitted = st.form_submit_button("Send")

    if submitted and user_input:
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("KrishnaAI is thinking..."):
            # Prepare Config with Thread ID
            # This ID will appear in LangSmith, allowing you to filter traces by specific user sessions
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Invoke Graph
            # All internal steps (Tool calls, LLM thoughts) are automatically sent to LangSmith
            response_state = graph.invoke(
                {"messages": [("user", user_input)]}, 
                config
            )
            
            final_response = response_state["messages"][-1].content
            
        # Add AI Message
        st.session_state.messages.append({"role": "AI", "content": final_response})
        st.rerun()