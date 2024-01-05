from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st

def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.error("OPENAI_API_KEY is not set")
        st.stop()
    else:
        st.success("OPENAI_API_KEY is set")

    st.title("Chat with us 📈")

    # Set the path to your specific file
    specific_file_path =  r"C:\Users\srula\OneDrive\Desktop\Medical Claims summarization using GenAI\patientdata.csv"

    # Check if the file exists
    if not os.path.isfile(specific_file_path):
        st.error("The specified file does not exist.")
        st.stop()

    with open(specific_file_path, 'rb') as file:
        agent = create_csv_agent(OpenAI(temperature=0), file, verbose=True)

    st.subheader("Ask a question about your CSV")
    user_question = st.text_input("Type your question here: ")

    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            st.write(agent.run(user_question))

if __name__ == "__main__":
    st.set_page_config(page_title="Chat with us")
    main()
