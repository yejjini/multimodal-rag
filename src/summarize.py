"""요약 체인: 텍스트·표는 LLM, 이미지는 VLM으로 검색용 요약문을 만든다."""

import base64
import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

MODEL = "gpt-4o-mini"

# 원문이 한국어 공시 문서이므로 텍스트·표 요약 지시문은 한국어로 둔다.
TEXT_PROMPT_KOR = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.
이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해주세요.간결하게.
요약할 표 또는 텍스트: {element}"""

# 이미지 지시문은 영어(토큰 절약, 지시 이행 품질), 출력만 한국어로 받는다.
IMAGE_PROMPT = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Provide a concise summary of the image that is well optimized for retrieval.
The summary should be written in Korean (Hangul)."""


def generate_text_summaries(texts, tables, summarize_texts=False):
    """텍스트·표 요약 리스트를 (text_summaries, table_summaries) 로 반환한다."""
    prompt = ChatPromptTemplate.from_template(TEXT_PROMPT_KOR)
    model = ChatOpenAI(temperature=0, model=MODEL)
    chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []
    if texts and summarize_texts:
        text_summaries = chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    table_summaries = chain.batch(tables, {"max_concurrency": 5}) if tables else []
    return text_summaries, table_summaries


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_summarize(img_base64, prompt=IMAGE_PROMPT):
    chat = ChatOpenAI(model=MODEL, max_tokens=1024)
    msg = chat.invoke([
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
        ])
    ])
    return msg.content


def generate_img_summaries(path):
    """폴더의 이미지마다 (Base64 원본, 요약문)을 같은 순서로 만든다."""
    img_base64_list, image_summaries = [], []
    for img_file in sorted(os.listdir(path)):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            b64 = encode_image(os.path.join(path, img_file))
            img_base64_list.append(b64)
            image_summaries.append(image_summarize(b64))
    return img_base64_list, image_summaries
