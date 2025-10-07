import streamlit as st
import requests
import uuid

API_BASE_URL = "http://backend:8000"


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🚀 DeepTrace Client")

with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDFs to the backend",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if st.button("Process Uploaded Files"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Sending files to backend..."):
                try:
                    files_to_send = [
                        ("files", (file.name, file.getvalue(), file.type))
                        for file in uploaded_files
                    ]

                    response = requests.post(f"{API_BASE_URL}/upload", files=files_to_send)

                    if response.status_code == 200:
                        st.success("Backend processed files successfully!")
                        st.json(response.json())
                    else:
                        st.error(f"Error from backend: {response.status_code} - {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to backend: {e}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask a question about your documents...")
if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Assistant is thinking..."):
            try:
                payload = {
                    "question": user_question,
                    "thread_id": st.session_state.thread_id
                }
                
                response = requests.post(f"{API_BASE_URL}/query", json=payload)

                if response.status_code == 200:
                    response_data = response.json()
                    answer = response_data.get("answer", "Sorry, I couldn't get an answer.")
                    
                    st.session_state.thread_id = response_data.get("thread_id")
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Backend Error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: Could not reach the backend. Is it running? Details: {e}")