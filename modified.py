from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
import requests
from io import StringIO

def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key is None or openai_api_key == "":
        st.error("OPENAI_API_KEY is not set")
        st.stop()
    else:
        st.success("OPENAI_API_KEY is set")
    
    st.title("Chat with us 📈")

    # Raw URL of the CSV file in your GitHub repository
    github_raw_csv_url = "patientdata.csv"

    # Fetch the CSV data from GitHub
    try:
        response = requests.get(github_raw_csv_url)
        if response.status_code == 200:
            # Read CSV data into a Pandas DataFrame
            df = pd.read_csv(StringIO(response.text))
            st.write(df)  # Display the DataFrame using Streamlit
        else:
            st.error("Failed to fetch the file. Please check the URL.")
            st.stop()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    # Create agent from CSV data
    agent = create_csv_agent(OpenAI(temperature=0), df, verbose=True)

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
