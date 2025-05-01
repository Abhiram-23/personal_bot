import streamlit as st
import asyncio
import json
import threading

from braedenbot import BraedenBot

def call_with_timeout(func, args=(), kwargs={}, timeout=20):
    result = {}
    def target():
        try:
            result["value"] = func(*args, **kwargs)
        except Exception as e:
            result["error"] = str(e)
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError("OpenAI call timed out.")
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["value"]

if "bot" not in st.session_state:
    st.session_state.bot = BraedenBot()
if "chat" not in st.session_state:
    st.session_state.chat = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(
    page_title="BraedenBot AI",       
    page_icon="🤖",                   
    layout="wide",                    
    initial_sidebar_state="expanded"  
)
st.title("🚀 BraedenBot Documentation Assistant")
st.markdown("A RAG-powered assistant for Braeden documentation")

query = st.text_input("Ask your question:", placeholder="Ask me about Braeden...")

if query:
    def process_query(query):
        bot = st.session_state.bot
        chat = st.session_state.chat
        chat.clear()

        chat.append({
            "role": "system",
            "content": bot.system_prompt  # must include JSON instruction
        })

        history = {"user_query": query}
        chat.append({"role": "user", "content": query})

        try:
            context = bot.get_context_for_query(query)
        except Exception as e:
            st.error(f"❌ Failed to retrieve context: {str(e)}")
            return

        chat.append({
            "role": "assistant",
            "content": f"Relevant Braeden documentation context:\n\n{context}"
        })

        conversation_active = True
        current_step = None
        count = 0

        with st.spinner("⏳ Processing your query..."):
            while conversation_active and count < 10:
                count += 1

                try:
                    # Validate chat format
                    for m in chat:
                        if "role" not in m or "content" not in m:
                            raise ValueError("Malformed chat message")
                    print(len(chat),chat, type(chat))
                    response = call_with_timeout(
                        bot.client2.chat.completions.create,
                        kwargs={
                        "model":"gemini-2.0-flash",
                        "messages":chat,
                        "response_format":{"type": "json_object"}
                                },
                        timeout=20
                    )

                    response_content = response.choices[0].message.content
                    parsed_output = json.loads(response_content)

                    chat.append({"role": "assistant", "content": response_content})

                    step = parsed_output.get("step", "").lower()
                    if step != current_step:
                        current_step = step
                        st.write(parsed_output.get("content", "No content found"))

                    if step == "output":
                        history["response"] = parsed_output.get("content", "No content found")
                        st.session_state.chat_history.append(history)
                        conversation_active = False

                except json.JSONDecodeError:
                    st.error("❌ Error: Invalid JSON from LLM")
                    break
                except TimeoutError:
                    st.error("❌ Request timed out. Try again.")
                    break
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    break

    process_query(query)

# === Chat History Viewer ===
st.divider()
st.subheader("Conversation History")

for msg in st.session_state.chat_history[::-1]:
    user_msg = msg.get("user_query")
    bot_msg = msg.get("response")

    if user_msg:
        col1, col2 = st.columns([1, 5])
        with col2:
            st.markdown(
                f"""
                <div style="
                    background-color:#222222;
                    color:#FFFFFF;
                    padding: 12px 18px;
                    border-radius: 15px;
                    margin: 5px 0;
                    font-size: 16px;
                    text-align: right;
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
                ">
                    {user_msg} 👨🏻‍💻
                </div>
                """,
                unsafe_allow_html=True
            )

    if bot_msg:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(
                f"""
                <div style="
                    background-color: #393E46;
                    color: #E8E8E8;
                    padding: 12px 18px;
                    border-radius: 15px;
                    margin: 5px 0;
                    font-size: 16px;
                    text-align: left;
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
                ">
                    🤖 {bot_msg}
                </div>
                """,
                unsafe_allow_html=True
            )

