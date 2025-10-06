from typing import TypedDict,List,Annotated,Optional,Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SubQuestionAnswer(TypedDict):
    sub_question:str
    sub_answer:str
    context:str


class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]
    question: str
    subquestion: List[str]
    rag_sub_answers: List[SubQuestionAnswer]
    rag_summary: Optional[str]
    tavily_results: Optional[str]
    final_answer: str
    decision_path: str 


    
    
