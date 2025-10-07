import streamlit as st
import pickle
import os
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
import concurrent.futures
import time
from langgraph.checkpoint.memory import MemorySaver
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval import create_retrieval_chain
from langchain_anthropic import ChatAnthropic
from anthropic import RateLimitError
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv
from prompts import sub_question_prompt, prompt, final_summarizer,decision 
from database import ingest
from state import AgentState 
import uuid
load_dotenv()
# os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
# os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

# if "thread_id" not in st.session_state:
#     st.session_state.thread_id = str(uuid.uuid4())
#     st.session_state.checkpoint=MemorySaver()
#     st.session_state.config={'configurable':{'thread_id':st.session_state.thread_id}}


FAISS_PATH = 'vectorstore'
BM25_PATH = "bm25.pkl"


rag_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0) 
summarizer_model = ChatGroq(model='meta-llama/llama-4-scout-17b-16e-instruct', temperature=0)
grade_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0) 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
checkpointer = MemorySaver()

# if "messages" not in st.session_state:
#     st.session_state.messages = []


# with st.sidebar.expander(" Upload PDFs"):
#     uploaded_pdfs = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True ,key="pdf_uploader_unique")
#     if uploaded_pdfs:
#         for pdf in uploaded_pdfs:
#             ingest(pdf)


_hybrid_retriever = None

def get_hybrid_retriever(faiss_path=FAISS_PATH, bm25_path=BM25_PATH):
    """
    Loads and returns the hybrid retriever. Caches it after the first load.
    Raises FileNotFoundError if the vector store files don't exist.
    """
    global _hybrid_retriever
    if _hybrid_retriever is not None:
        return _hybrid_retriever

    if not os.path.exists(faiss_path) or not os.path.exists(bm25_path):
        raise FileNotFoundError("Vector store not found. Please upload documents before querying.")

    try:
        embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)

        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        _hybrid_retriever = EnsembleRetriever(retrievers=[semantic_retriever, bm25], weights=[0.6, 0.4])
        print("Hybrid retriever loaded successfully.")
        return _hybrid_retriever
    except Exception as e:
        raise RuntimeError(f"Failed to load retrievers: {e}")

# hybrid_retriever = get_hybrid_retriever()

# document_chain = create_stuff_documents_chain(
#     llm=rag_model,
#     prompt=PromptTemplate.from_template(prompt)
# )

# retriever_chain = create_retrieval_chain(
#     retriever=hybrid_retriever,
#     combine_docs_chain=document_chain
# )



def decompose_question(state: AgentState) -> AgentState:
    # with st.spinner('Decomposing question....'):
        question = state['question']
        prompt_chain = PromptTemplate.from_template(sub_question_prompt)
        chain = prompt_chain | rag_model | StrOutputParser()
        questions = chain.invoke({'question': question})
        state['subquestion'] = [q.strip() for q in questions.split("\n") if q.strip()]
        # st.success(f"QUESTION SUCCESSFULLY DECOMPOSED")
        return state


def retrieve_sub_answers_for_rag(state: AgentState) -> AgentState:
        try:
            hybrid_retriever = get_hybrid_retriever()
            document_chain = create_stuff_documents_chain(llm=rag_model, prompt=PromptTemplate.from_template(prompt))
            retriever_chain = create_retrieval_chain(retriever=hybrid_retriever, combine_docs_chain=document_chain)
        except FileNotFoundError as e:
            # This allows the API to catch the error and tell the user to upload a file
            raise e
    # with st.spinner('Retrieving sub-answers for RAG....'):
        results = []
        sub_qs = state['subquestion']
        def safe_invoke_rag(q: str):
            try:
                rag_result = retriever_chain.invoke({"input": q})
                answer = rag_result['answer']
                docs = rag_result['context']
                context = "\n".join([f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}"for doc in docs])
                return (q, answer,context)
            except Exception as e:

                # st.warning(f"ANSWER NOT FOUND REASON-> {str(e)}")
                return (q, f"Retrieval failed for: {q} - {str(e)}", "No context due to error")
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(safe_invoke_rag, q) for q in sub_qs]
            time.sleep(0.1)
            results = [f.result() for f in futures]

        state['rag_sub_answers'] = results
        # st.success("RAG SUB-ANSWERS COMPLETED")
        return state



def summarize_rag_answers(state: AgentState) -> AgentState:
    # with st.spinner('Summarizing RAG-only answers....'):

        summarizer_prompt = PromptTemplate.from_template(final_summarizer) 
        answers_context = "\n\n".join([f"Sub-question: {q}\nSub-answer: {a}\nContext:{context}" for q, a,context in state['rag_sub_answers']])

        summarization_chain = summarizer_prompt | summarizer_model | StrOutputParser()

        rag_summary = summarization_chain.invoke({
            'question': state['question'],
            'context': answers_context
        })

        state['rag_summary'] = rag_summary
        # st.success("RAG SUMMARY COMPLETED")
        return state


def decide_next_step(state: AgentState) -> AgentState: 
    # with st.spinner('Need for Online Search....'):
        question = state['question']
        rag_summary = state['rag_summary']
        decision_prompt = PromptTemplate.from_template(decision)
        decision_chain = decision_prompt | grade_llm | StrOutputParser()
        decision_output = decision_chain.invoke({'question': question,'rag_summary': rag_summary}).strip().upper()

        # st.write(f"LLM Decision: {decision_output}")

        if "YES_ONLINE_SEARCH" in decision_output:
            state['decision_path'] = "use_tavily"
        else:
            state['decision_path'] = "summarize_rag_only"
        
        return state

def perform_online_search(state: AgentState) -> AgentState:
    # with st.spinner('Performing Online-Search....'):
        question =state['question']
        try:
            tavily_tool = TavilySearchResults(max_results=3)
            tavily_results = tavily_tool.invoke({"query": question})

            state['tavily_results'] = tavily_results
            # st.success("Online search completed.")
        except Exception as e:
            # st.error(f"WEB SEARCH ERROR REASON-> {e}")

            state['tavily_results'] = f"Error during online search: {str(e)}"

        return state

def combine_all_answers(state: AgentState) -> AgentState:
    # with st.spinner('Making bried Report....'):
        question = state['question']
        rag_summary = state['rag_summary']
        tavily_results = state.get('tavily_results', None) 

        


        history = state['messages'][:-1]
        chat_history = "\n".join(
            [f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in history]
        )


        # st.write("chat_history",state['messages'])


    

        # for msg in st.session_state.messages:
        #     chat_history.append(f"{msg['role']}:{msg['content']}")
        # chat_history = "\n".join(chat_history)

        if tavily_results is None:
            tavily_content = "No Tavily results available."
        
        else:
            tavily_content="\n\n".join(f"Title:{item['title']}\nUrl:{item['url']}\nContent:{item['content'][:200]}" for item in tavily_results)

        tavily_content=str(tavily_content)

        combined_context = f"=== INTERNAL DOCUMENTS ===\n{rag_summary}\n\n"
        combined_context += f"Online Search Results: {tavily_content}\n\n"

        combined_context +=f"=== Previous Knowledge ===: {chat_history}"

        summarization_prompt = PromptTemplate.from_template(final_summarizer)
        summarization_chain = summarization_prompt | summarizer_model | StrOutputParser()

        try:
            final_answer = summarization_chain.invoke({'question': question,'context': combined_context})

            state['final_answer'] = final_answer

            

            state['messages'].append(AIMessage(content=final_answer))
            
            # st.write("STATE MESSAGES",state['messages'])
            # st.success("REPORT GENERATED")
            return state
        except Exception as e:
            # st.error(f"ERROR REASON-> {e}")
            state['final_answer'] = f"An error occurred during final answer combination: {str(e)}"
            state['messages'].append(AIMessage(content=final_answer))

        return state

def route_based_on_decision(state: AgentState) -> str:
    """Reads the decision_path from state to route the graph."""
    return state['decision_path']


from state import AgentState
from langgraph.graph import StateGraph, END,START


def build_multi_source_rag_graph():
    graph = StateGraph(AgentState)

    graph.add_node("decompose_question", decompose_question)
    graph.add_node("retrieve_sub_answers_for_rag", retrieve_sub_answers_for_rag)
    graph.add_node("summarize_rag_answers", summarize_rag_answers)
    graph.add_node("decide_next_step", decide_next_step)
    graph.add_node("perform_online_search", perform_online_search)
    graph.add_node("combine_all_answers", combine_all_answers)

    graph.add_edge(START,'decompose_question')

    graph.add_edge("decompose_question", "retrieve_sub_answers_for_rag")
    graph.add_edge("retrieve_sub_answers_for_rag", "summarize_rag_answers")
    graph.add_edge("summarize_rag_answers", "decide_next_step") 

    graph.add_conditional_edges(
        "decide_next_step",
        route_based_on_decision, 
        {
            "use_tavily": "perform_online_search",
            "summarize_rag_only": "combine_all_answers",
        }
    )

    graph.add_edge("perform_online_search", "combine_all_answers")

    graph.add_edge("combine_all_answers", END)

    return graph.compile(checkpointer=checkpointer)

app = build_multi_source_rag_graph()


# st.title(' DeepTrace ')

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# user_question = st.chat_input("Ask me anything...")

# if user_question:
#     with st.chat_message("user"):
#         st.markdown(user_question)

#     final_answer=None
#     with st.expander('Thinking'):
#         with st.chat_message("assistant"):
#             with st.spinner("Processing your question..."):
#                 try:
#                     config = {
#                         "configurable": {
#                             "thread_id": st.session_state.thread_id,
#                         }
#                     }

#                     initial_graph_state = {"question":user_question,
#                                                     "subquestion":[],
#                                                     "rag_sub_answers":[],
#                                                     "rag_summary":None,
#                                                     "tavily_results":None,
#                                                     "final_answer":"",
#                                                     "decision_path":"",
#                                                     "messages":HumanMessage(content=user_question)}
                    
                    
#                     # Pass the config to the invoke method
#                     response = app.invoke(initial_graph_state, config=config)


#                     final_answer = response['final_answer']

#                 except Exception as e:
#                     st.error(f" An unexpected error occurred: {e}")
#                     st.exception(e)

               
#                 except Exception as e:
#                     st.error(f" An unexpected error occurred: {e}")
#                     st.exception(e)


#     if final_answer:
#         st.session_state.messages.append({"role": "user", "content": user_question})
#         st.markdown(final_answer)
#         st.session_state.messages.append({"role": "assistant", "content": final_answer})


# # [
# # 0:"HumanMessage(content='who am i', additional_kwargs={}, response_metadata={}, id='076622e1-e263-4f2f-affb-cfa7d527a785')"
# # ]