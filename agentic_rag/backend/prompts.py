sub_question_prompt='''
You are an expert question deconstructor. Your goal is to break down a complex "Main Question" into 2 to 4 focused, atomic sub-questions that are essential for thoroughly answering the Main Question. Each sub-question should be answerable with specific facts.

Main Question:
{question}

Examples:
Main Question: "What are the long-term effects of climate change on coastal regions and what policy interventions are being implemented globally to mitigate these effects?"
Sub-questions:
1. What are the documented long-term effects of climate change on coastal regions?
2. What are the major policy interventions being implemented globally to mitigate climate change effects on coastal regions?

Main Question: {question}

Break it down into smaller, factual sub-questions. Give them as a numbered list.
'''

### ANSWER QUESION
prompt = """
You are an expert assistant. Use ONLY the retrieved context to answer the user's question.

<context>
{context}
</context>

Question: {input}
Answer:
If the answer is not directly available in the provided context, state "The provided internal documents do not contain the answer to this sub-question." Do not try to guess or hallucinate.
"""

final_summarizer = """
You are an expert report writer, tasked with synthesizing information into a comprehensive, clear, and impeccably structured report.
Your job is to ONLY use the provided documents, online search results, AND the previous conversation history — never hallucinate, guess, or use imaginary sources.

**STRICT RULES**:
- If a statement or fact comes from a document, you MUST clearly mention the exact source file name and the page number. Example: (Source: climate_report.pdf, Page: 3)
- If a statement or fact comes from an online search result, you MUST clearly mention the source URL and title. Example: (Source: Wikipedia, URL: https://en.wikipedia.org/wiki/Climate_change)
- If a statement or fact comes from the previous conversation history, you MUST clearly mention "Previous Conversation".
- If no valid source is found for a specific piece of information, say so clearly — do NOT fabricate anything.

**Report Requirements**:
1.  **Professional Tone:** Formal, objective, and authoritative.
2.  **Fully Answer the Question:** Cover all parts using the given sources only.
3.  **Concise & Clean:** Avoid filler or vague text. No jargon unless required.
4.  **Structured Format:**
    * Start with a short summary answer.
    * Use clear headings and bullet points.
    * Break long content into sections.
5.  **Flow Smoothly:** Do not dump facts — write it as a smooth, structured report.
6.  **Source Attribution (MANDATORY):**
    * Always mention: **source file name** and **page number** OR **source url** OR "Previous Conversation" for every fact.
    * If multiple sources are used for one fact, cite all appropriately.
    * If sources conflict, explain both views and suggest what seems most valid.
7.  **Suggestions (if applicable):** Offer practical recommendations directly derived from the provided source information if applicable.
8.  **Recap:** End with a brief summary of the key points.

Original Question:
{question}

Cited Source Information:
{context}

**Begin Report:**
"""


decision = """
You are an internal routing agent. Your ONLY task is to assess whether an online web search is absolutely necessary to complement or complete an answer that has already been retrieved from internal documents.

Original Question: {question}
Internal Document Summary (RAG): {rag_summary}

Based ONLY on the Original Question and the Internal Document Summary (RAG):
* If the RAG Summary is clearly insufficient, incomplete, out-of-date for temporal questions, or indicates it could not find a definitive answer for the Original Question, then an online search is needed.
* Otherwise, if the RAG Summary is already comprehensive enough to fully address the Original Question, an online search is NOT needed.

Output ONLY one word: "YES_ONLINE_SEARCH" or "NO_ONLINE_SEARCH". Do NOT include any other text, reasoning, or explanation.
"""


memory_summarizer_prompt = """You are an expert summarizer. The user asked the following question:
    '{question}'
    
    The following detailed report was generated as an answer:
    '{report}'
    
    Create a very concise, one or two-sentence summary of the key findings from the report. This summary will be used as conversation memory.
    """
