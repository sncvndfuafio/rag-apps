# rag_app/api/routes_chat.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END

from chunking.vectordb_service import vectordb_service
from core.config import config

router = APIRouter()

# --- LangGraph Setup ---
class GraphState(BaseModel):
    """
    Represents the state of our graph.
    """
    question: str
    context: str = ""
    answer: str = ""

def retrieve(state: GraphState) -> GraphState:
    print("---RETRIEVE---")
    question = state.question
    retriever = vectordb_service.get_retriever()
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])
    return GraphState(question=question, context=context)

def generate(state: GraphState) -> GraphState:
    print("---GENERATE---")
    question = state.question
    context = state.context

    llm = ChatGroq(
        temperature=0,
        groq_api_key=config.GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile" # Using the smaller, generally stable Llama 3 for now
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are an assistant for question-answering tasks.
        Use the following retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Keep the answer concise and relevant to the provided context.

        Question: {question}
        Context: {context}
        Answer:
        """
    )

    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke({"question": question, "context": context})
    return GraphState(question=question, context=context, answer=answer)

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.set_entry_point("retrieve")
app = workflow.compile()
# --- End LangGraph Setup ---

class ChatQuery(BaseModel):
    query: str

@router.post("/chat", response_model=Dict[str, str])
async def chat(chat_query: ChatQuery):
    inputs = {"question": chat_query.query}
    result = app.invoke(inputs)
    # FIX: Access the answer from the 'result' dictionary using key access
    return {"answer": result["answer"]}