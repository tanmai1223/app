from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import re

# Set your OpenAI API key directly
openai_api_key = "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"


def extract_demographics(summary_text):
    # Extracting Patient Demographics using regular expressions
    patient_demographics = re.search(r'Patient Demographics: (.+?)\n', summary_text)
    if patient_demographics:
        return patient_demographics.group(1)
    return "No demographics found."

def process_large_input(user_question, agent):
    # Split the user input into smaller segments (adjust segment length as needed)
    segment_length = 100
    segments = [user_question[i:i+segment_length] for i in range(0, len(user_question), segment_length)]

    results = []
    for segment in segments:
        result = agent.run(segment)
        results.append(result)

    return results

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
    specific_file_path = os.path.join(os.path.dirname(__file__), "patientdata.csv")

    # Check if the file exists
    if not os.path.isfile(specific_file_path):
        st.error("The specified file does not exist.")
        st.stop()

    with open(specific_file_path, 'rb') as file:
        agent = create_csv_agent(OpenAI(temperature=0), file, verbose=True)

    st.subheader("Ask a question about your CSV")
    user_question = st.text_area("Type your question here: ", height=200)

    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            # Process large input segments and collect results
            results = process_large_input(user_question, agent)

            # Aggregate results
            aggregated_summary = ""
            for result in results:
                if 'summary' in result:
                    aggregated_summary += result['summary']

            # Extracting information from the aggregated summary
            if aggregated_summary:
                demographics_info = extract_demographics(aggregated_summary)
                st.write("Patient Demographics:")
                st.write(demographics_info)
            else:
                st.write("No summary found in the result.")

if __name__ == "__main__":
    st.set_page_config(page_title="Chat with us")
    main()
