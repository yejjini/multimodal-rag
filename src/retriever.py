"""다중 벡터 검색기 + 최종 체인.

요약문(검색용)은 Chroma 벡터 DB에, 원본(답변용)은 InMemoryStore에 넣고
같은 doc_id 로 연결한다. 질문은 요약으로 찾고, 답변은 원본으로 만든다.
"""

import re
import uuid

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
ID_KEY = "doc_id"
B64_PATTERN = re.compile(r"^[A-Za-z0-9+/]{100,}={0,2}$")

ANSWER_INSTRUCTION = (
    "아래 자료를 근거로 질문에 한국어로 답하세요.\n"
    "자료에 없는 내용은 모른다고 답하세요.\n\n"
)


def build_multivector_retriever(collection_name="multimodal_rag"):
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    return MultiVectorRetriever(
        vectorstore=vectorstore, docstore=InMemoryStore(), id_key=ID_KEY
    )


def add_documents(retriever, doc_summaries, doc_contents):
    """요약은 vectorstore 에, 원본은 docstore 에 같은 id 로 저장한다."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={ID_KEY: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


def is_image(content: str) -> bool:
    return bool(B64_PATTERN.match(content))


def split_image_text_types(docs):
    """검색 결과를 Base64 이미지와 텍스트로 나눈다."""
    images, texts = [], []
    for doc in docs:
        content = doc.page_content if isinstance(doc, Document) else doc
        (images if is_image(content) else texts).append(content)
    return {"images": images, "texts": texts}


def img_prompt_func(data_dict):
    """이미지를 먼저, 그다음 텍스트 근거와 질문을 하나의 멀티모달 메시지로 조립한다."""
    messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        for img in data_dict["context"]["images"]
    ]
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append({
        "type": "text",
        "text": ANSWER_INSTRUCTION
        + f"질문: {data_dict['question']}\n\n참고 텍스트:\n{formatted_texts}",
    })
    return [HumanMessage(content=messages)]


def build_multimodal_chain(retriever):
    model = ChatOpenAI(temperature=0, model=CHAT_MODEL, max_tokens=1024)
    return (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )


def build_text_only_chain(text_chunks, k=4, collection_name="text_only_rag"):
    """비교용: 이미지·표 없이 원본 텍스트 청크만 임베딩한 일반 RAG."""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    vectorstore.add_texts(texts=text_chunks)
    text_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_template(
        ANSWER_INSTRUCTION + "질문: {question}\n\n참고 텍스트:\n{context}"
    )
    model = ChatOpenAI(temperature=0, model=CHAT_MODEL, max_tokens=1024)
    return (
        {
            "context": text_retriever
            | RunnableLambda(lambda docs: "\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )
