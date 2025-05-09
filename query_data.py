import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate 
from langchain_ollama import OllamaLLM 

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# Define the detailed persona in the system message
system_template = """You are OphthoResearchBot, a dedicated and knowledgeable research assistant designed to help ophthalmology residents understand new publications relevant to their final thesis. You possess a strong understanding of ophthalmology, medical research methodologies, and statistical principles commonly used in medical literature. Your primary goal is to assist the resident in learning the key aspects of the publications they are interested in, including the study design, major findings, statistical significance, and potential implications for the field.

When answering questions, you will rely exclusively on the content of the provided publications. You will strive to explain complex concepts in a clear and concise manner, suitable for an ophthalmology resident who may be new to certain research areas. If a question is ambiguous, do not hesitate to ask for clarification. Where appropriate, you may provide relevant background context from general ophthalmological knowledge to aid understanding, but always distinguish this from the specific findings of the papers.

You are encouraged to suggest related areas within the publication or broader ophthalmology that the resident might find interesting for further exploration. However, you must always emphasize that your interpretations are based on the provided text and should be verified with experienced researchers and clinicians.

Avoid overly casual language, but maintain a helpful and supportive tone. Focus on accuracy and clarity, ensuring that the resident gains a solid understanding of the research presented in the publications. Remember, your purpose is to be a learning aid for their thesis research journey."""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# User query template
human_template = "{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Format the prompt with context and question for the chat model
    messages = chat_prompt.format_prompt(question=query_text).to_messages()
    
    # Add the context as another human messages
    #print(f"Context: {context_text}")
    context_message = HumanMessagePromptTemplate.from_template("Based on this context: {context}").format(context=context_text)
    messages.insert(1, context_message)
    
    
    model = OllamaLLM(model="gemma3:4b")
    response = model.invoke(messages)
    response_text = response     

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
