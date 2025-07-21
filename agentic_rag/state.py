from typing import TypedDict,List,Annotated,Optional

class SubQuestionAnswer(TypedDict):
    sub_question:str
    sub_answer:str
    context:str


class AgentState(TypedDict):
    question: str
    subquestion: List[str]
    rag_sub_answers: List[SubQuestionAnswer]
    rag_summary: Optional[str]
    tavily_results: Optional[str]
    final_answer: str
    decision_path: str # NEW: To store the decision ("use_tavily" or "summarize_rag_only")


    
    
