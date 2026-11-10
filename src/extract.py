"""PDF 파싱 함수: PDF 한 편을 텍스트 / 표 / 이미지로 분리한다."""

from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf


def extract_pdf_elements(path: str, fname: str):
    """PDF에서 텍스트 블록·표·이미지를 추출한다. 이미지는 path/figures/ 에 저장된다.

    hi_res 전략은 페이지를 이미지로 보고 레이아웃을 분석하기 때문에
    fast 전략이 놓치는 표 구조와 그림 영역까지 잡아낸다 (대신 느리다).
    """
    return partition_pdf(
        filename=path + fname,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=path + "figures",
    )


def categorize_elements(raw_pdf_elements):
    """추출 요소를 (texts, tables) 두 리스트로 분류한다."""
    texts, tables = [], []
    for element in raw_pdf_elements:
        kind = str(type(element))
        if "unstructured.documents.elements.Table" in kind:
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in kind:
            texts.append(str(element))
    return texts, tables


def split_texts(texts, chunk_size: int = 2000, chunk_overlap: int = 200):
    """텍스트 블록을 토큰 기준으로 다시 자른다. overlap 은 경계에서 문맥이 끊기는 것을 막는다."""
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(" ".join(texts))
