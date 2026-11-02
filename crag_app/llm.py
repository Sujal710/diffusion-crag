"""Groq LLMs plus every prompt the CRAG pipeline uses."""
from functools import lru_cache
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from .config import get_settings


@lru_cache
def generator_llm() -> ChatGroq:
    s = get_settings()
    return ChatGroq(model=s.groq_generator_model, api_key=s.groq_api_key, temperature=0.2)


@lru_cache
def grader_llm() -> ChatGroq:
    s = get_settings()
    return ChatGroq(model=s.groq_grader_model, api_key=s.groq_api_key, temperature=0)


# ---------------------------------------------------------------------------
# 1. Retrieval evaluator: how relevant is ONE document to the question?
# ---------------------------------------------------------------------------
class RelevanceScore(BaseModel):
    score: float = Field(description="Relevance from 0.0 (useless) to 1.0 (directly answers it)")
    reason: str = Field(description="One short sentence explaining the score")


relevance_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a retrieval evaluator. Score how useful the document is for answering "
     "the question. 1.0 = contains the answer, 0.5 = related but incomplete or possibly "
     "outdated, 0.0 = unrelated or wrong."),
    ("human", "Question: {question}\n\nDocument:\n{document}"),
])


def relevance_chain():
    return relevance_prompt | grader_llm().with_structured_output(RelevanceScore)


# ---------------------------------------------------------------------------
# 2. Knowledge refinement: which small strips are worth keeping?
# ---------------------------------------------------------------------------
class StripSelection(BaseModel):
    keep: List[int] = Field(description="Numbers of the strips that help answer the question")


strip_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You filter knowledge. Below are numbered text strips. Return ONLY the numbers of "
     "strips that contain facts useful for answering the question. Drop filler, "
     "unrelated or redundant strips."),
    ("human", "Question: {question}\n\nStrips:\n{strips}"),
])


def strip_filter_chain():
    return strip_prompt | grader_llm().with_structured_output(StripSelection)


# ---------------------------------------------------------------------------
# 3. Query rewriter for web search (CRAG "Incorrect" path)
# ---------------------------------------------------------------------------
class SearchQuery(BaseModel):
    query: str = Field(description="Short keyword-style web search query")


web_query_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the question into a concise web search query (max ~10 words)."),
    ("human", "{question}"),
])


def web_query_chain():
    return web_query_prompt | grader_llm().with_structured_output(SearchQuery)


# ---------------------------------------------------------------------------
# 4. Generator
# ---------------------------------------------------------------------------
generate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question using ONLY the knowledge provided. Be concise and factual. "
     "If the knowledge is not enough, say exactly what is missing instead of guessing."),
    ("human", "Question: {question}\n\nKnowledge:\n{knowledge}"),
])


def generate_chain():
    return generate_prompt | generator_llm()


# ---------------------------------------------------------------------------
# 5. Final answer checker (the extra loop on top of CRAG)
# ---------------------------------------------------------------------------
class AnswerCheck(BaseModel):
    is_correct: bool = Field(description="True if the answer fully and correctly answers the question")
    feedback: str = Field(description="If not correct: what is wrong or missing")


check_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict answer checker. Decide whether the answer correctly and completely "
     "answers the question, and is supported by the knowledge. Mark it incorrect if it "
     "is off-topic, vague, contradicts the knowledge, or says the information is missing."),
    ("human", "Question: {question}\n\nKnowledge:\n{knowledge}\n\nAnswer:\n{answer}"),
])


def check_chain():
    return check_prompt | grader_llm().with_structured_output(AnswerCheck)


# ---------------------------------------------------------------------------
# 6. Question reframer (runs when the checker says the answer is wrong)
# ---------------------------------------------------------------------------
class ReframedQuestion(BaseModel):
    question: str = Field(description="Clearer, more specific version of the question")


reframe_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "The previous attempt to answer this question failed. Rewrite the question so that "
     "retrieval finds better documents: make it more specific, add likely keywords, and "
     "fix ambiguity. Keep the user's original intent. Do not answer it."),
    ("human",
     "Original question: {original_question}\n"
     "Question used last time: {question}\n"
     "Checker feedback: {feedback}"),
])


def reframe_chain():
    return reframe_prompt | grader_llm().with_structured_output(ReframedQuestion)


# ---------------------------------------------------------------------------
# 8. Storyboard: turn one idea into N consistent image prompts (text -> video)
# ---------------------------------------------------------------------------
class Storyboard(BaseModel):
    style: str = Field(description="One sentence of visual style and recurring subjects, reused for every frame")
    scenes: List[str] = Field(description="One image prompt per frame, in time order")


storyboard_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You turn an idea into a storyboard for a short video made of still images. Write one "
     "shared 'style' sentence (art style, lighting, colours, and a fixed description of any "
     "recurring character or place so they look the same in every frame). Then write exactly "
     "{n} scene prompts in time order, each describing ONE moment: subject, action, camera "
     "angle. Each scene should change only a little from the previous one so the video flows. "
     "Prompts must be visual descriptions only, with no text, captions or dialogue."),
    ("human", "Idea: {idea}"),
])


def storyboard_chain():
    return storyboard_prompt | generator_llm().with_structured_output(Storyboard)


# ---------------------------------------------------------------------------
# 7. Knowledge-base summary (used by the UI's "Summarise contents" button)
# ---------------------------------------------------------------------------
kb_summary_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You describe what a document collection contains. You are given a few snippets "
     "from each document. For each document write one line on what it is about, then "
     "give a short overall summary of the topics covered. Only use the snippets; if they "
     "are too thin to tell, say so. Keep it brief."),
    ("human", "{documents}"),
])


def kb_summary_chain():
    return kb_summary_prompt | generator_llm()
