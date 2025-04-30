import streamlit as st
from braedenbot import BraedenBot
import json
import asyncio

# Initialize the app
st.set_page_config(page_title="BraedenBot CLI to Streamlit", layout="wide")
st.title("🛠️ BraedenBot — Documentation Assistant")

# Custom CSS for chat bubbles
st.markdown("""
<style>
.chat-bubble {
    border-radius: 1rem;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    max-width: 80%;
}
.user-msg {
    background-color: #e6f0ff;
    border-left: 4px solid #3399ff;
    margin-left: 20%;
}
.assistant-msg {
    background-color: #f2f2f2;
    border-left: 4px solid #999999;
    margin-right: 20%;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = BraedenBot()
    st.session_state.messages = [{"role": "system", "content": st.session_state.bot.system_prompt}]
    st.session_state.awaiting_response = False

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        try:
            parsed = json.loads(message["content"])
            content = parsed.get("content", message["content"])
        except json.JSONDecodeError:
            content = message["content"]
        with st.chat_message("assistant"):
            st.markdown(content)

# Input area
query = st.chat_input("Ask your question about Braeden docs", key="user_input")

# Handle new user query
if query and not st.session_state.awaiting_response:
    try:
        context = st.session_state.bot.get_context_for_query(query)
        # Store user query and context
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": f"Relevant Braeden documentation context:\n\n{context}"})

        # Trigger final LLM response generation
        with st.spinner("Thinking..."):
            response = st.session_state.bot.client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=st.session_state.messages,
            )
            response_content = response.choices[0].message.content
            print("response_content",response_content)
            try:
                parsed_output = json.loads(response_content)
                step = parsed_output.get("step", "").lower()
                if step == "output":
                    final_answer = parsed_output.get("content", "No answer found.")
                    print("final_answer", final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    with st.chat_message("assistant"):
                        st.markdown(final_answer)
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON response from Gemini")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.messages = [{"role": "system", "content": st.session_state.bot.system_prompt}]
    st.session_state.awaiting_response = False
    st.rerun()