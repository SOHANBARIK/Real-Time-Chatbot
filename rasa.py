import streamlit as st
import os
import groq  # <--- Import groq to handle errors
import pytz  # <--- NEW: Required for global timezones
from dotenv import load_dotenv

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()

# Optional: verify tracing
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

st.set_page_config(page_title="AI Chatbot", layout="centered")

# --- 2. DEFINE TOOLS & GRAPH ---

@tool
def get_world_time(timezone_str: str = "UTC"):
    """
    Get the current time for a specific timezone. 
    The input should be a valid pytz timezone string (e.g., 'America/New_York', 'Asia/Kolkata', 'Europe/London').
    If the user asks for a city, convert it to the correct timezone string first.
    """
    try:
        # 1. Load the specific timezone
        tz = pytz.timezone(timezone_str)
        
        # 2. Get current time in that zone
        now = datetime.now(tz)
        
        # 3. Return a readable string
        return now.strftime(f"%Y-%m-%d %I:%M:%S %p ({timezone_str})")
        
    except pytz.UnknownTimeZoneError:
        return f"Error: '{timezone_str}' is not a valid timezone. Please try a standard format like 'Asia/Tokyo' or 'America/Chicago'."

@tool
def get_weather(city: str):
    """Get current weather using Open-Meteo."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_response = requests.get(geo_url).json()
        if not geo_response.get("results"):
            return f"Could not find coordinates for {city}."
        location = geo_response["results"][0]
        lat, lon = location["latitude"], location["longitude"]
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
    # Check Critical Keys
    if not os.getenv("GROQ_API_KEY"):
        st.error("🚨 GROQ_API_KEY is missing in .env")
        st.stop()
    if not os.getenv("TAVILY_API_KEY"):
        st.error("🚨 TAVILY_API_KEY is missing in .env")
        st.stop()

    # 1. Define Tools (Updated to use get_world_time)
    tools = [TavilySearchResults(max_results=2), get_world_time, get_weather]

    # 2. Initialize LLM (Dynamic Model Name)
    model_name = os.getenv("GROQ_MODEL_NAME")
    
    llm = init_chat_model(model_name, model_provider='groq', temperature=0)
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

    # 6. Compile
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

graph = get_graph()

# --- 3. SESSION STATE ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- 4. UI STYLING ---
st.markdown(
    """
    <style>
        body { background-color:rgb(1, 1, 1); color: white; font-family: 'Arial', sans-serif; }
        .stApp { background-color:rgb(1, 1, 1); }
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
        .message-box { padding: 15px; border-radius: 10px; margin: 10px 0; }
        .user-message { background-color:rgb(153, 150, 242); text-align: right; border: 1px solid #ffffff; color: black; }
        .ai-message { background-color: #ffffff; text-align: left; color: black; border: 1px solid #ffffff; }
        .stTextInput input { background-color: #ffffff; color: black; border: 1px solid #444; }
    </style>
    <div class="fixed-top-left">AI Chatbot (LangSmith Active)</div>
    <div class="fixed-bottom">Made with ❤️ from Sohan</div>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: white;'>What can I help with?</h1>", unsafe_allow_html=True)

# --- 5. CHAT LOGIC ---
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "ai-message"
        st.markdown(f"<div class='message-box {role_class}'>{message['content']}</div>", unsafe_allow_html=True)

with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("", placeholder="Ask me anything...", key="chat_input_form", label_visibility="visible")
    submitted = st.form_submit_button("Send")

    if submitted and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("AI is thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            try:
                # --- EXECUTION WITH ERROR HANDLING ---
                response_state = graph.invoke(
                    {"messages": [("user", user_input)]}, 
                    config
                )
                final_response = response_state["messages"][-1].content
                st.session_state.messages.append({"role": "AI", "content": final_response})
                st.rerun()

            except groq.RateLimitError:
                st.error("⏳ Rate Limit Hit! Please wait a moment or switch to the 8b model in .env.")
            except Exception as e:
                st.error(f"An error occurred: {e}")