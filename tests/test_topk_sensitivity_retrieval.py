import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retriever.run_topk_sensitivity_retrieval import (
    build_query_texts,
    format_retrieved_docs,
)


def test_build_query_texts_for_question_direct_and_hyde_modes():
    questions = [
        {"id": "q1", "question": "Who wrote the novel?"},
        {"id": "q2", "question": "Where was the actor born?"},
    ]
    answers_by_id = {
        "q1": {"id": "q1", "question": "Who wrote the novel?", "prediction": "novel author"},
        "q2": {"id": "q2", "question": "Where was the actor born?", "prediction": ""},
    }

    assert build_query_texts(questions, {}, "question_only") == [
        "Who wrote the novel?",
        "Where was the actor born?",
    ]
    assert build_query_texts(questions, answers_by_id, "direct_rewrite") == [
        "novel author",
        "Where was the actor born?",
    ]
    assert build_query_texts(questions[:1], answers_by_id, "hyde") == [
        "Who wrote the novel?\n\nHypothetical supporting passage:\nnovel author"
    ]


def test_format_retrieved_docs_keeps_scores_and_doc_metadata():
    docs = [
        {
            "doc_id": "d0",
            "title": "Alpha",
            "text": "alpha text",
            "source_question_id": "q0",
        },
        {
            "doc_id": "d1",
            "title": "Beta",
            "text": "beta text",
            "source_question_id": "q1",
        },
    ]

    retrieved = format_retrieved_docs([0.8, 0.7], [1, 0], docs)

    assert retrieved == [
        {
            "score": 0.8,
            "doc_id": "d1",
            "title": "Beta",
            "text": "beta text",
            "source_question_id": "q1",
        },
        {
            "score": 0.7,
            "doc_id": "d0",
            "title": "Alpha",
            "text": "alpha text",
            "source_question_id": "q0",
        },
    ]
