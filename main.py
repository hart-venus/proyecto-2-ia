from RagApi import RagApi
from langchain.prompts import PromptTemplate  
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import csv

def format_output(question, rag_result, llm_result):
    output = f"Question: {question}\n\n"
    output += "RAG API Result:\n"
    output += f"{rag_result}\n\n"
    output += "LLM Result:\n"
    output += f"{llm_result}\n"
    output += "-" * 50 + "\n"
    return output

def create_chain(llm): 
        template = """
            Question: {question}
            Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        return LLMChain(
            llm=llm, 
            prompt=QA_CHAIN_PROMPT
        )

if __name__ == '__main__':
    ra = RagApi(load_vectorstore=True)  # change this line to build from scratch
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # TODO: use stock llama 3
    nonragchain = create_chain(llm)

    questions = [
        "How do I get a job at Google?",
        "What are the key elements of a strong resume?",
        "How can I prepare for a technical interview?",
        "What are some tips for negotiating salary?",
        "How can I improve my work-life balance?",
        "How can I showcase my problem-solving skills on my resume?",
        "What are some effective ways to highlight my technical skills on my resume?",
        "How can I quantify my achievements and impact on my resume?",
        "How can I showcase my leadership and teamwork skills on my resume?",
        "How can I make my resume stand out among a large pool of applicants?",
    ]

    results = []

    for question in questions:
        rag_result = ra.chain({"query": question})["result"]
        llm_result = nonragchain({"question": question})["text"]

        results.append([question, rag_result, llm_result])

        print(format_output(question, rag_result, llm_result))

    # Export results to a CSV file
    with open("results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Question", "RAG API Result", "LLM Result"])
        writer.writerows(results)

