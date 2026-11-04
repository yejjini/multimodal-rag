# Multimodal RAG

PDF의 텍스트·표·이미지를 함께 활용하는 멀티모달 RAG 파이프라인.

## 구조

```
multimodal-rag/
├── data/            # 원본 PDF, 추출 이미지(figures/)
├── notebooks/       # 실습 노트북
└── src/
    ├── extract.py   # PDF 파싱
    ├── summarize.py # 요약 체인
    └── retriever.py # 다중 벡터 검색기
```

## 실행

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt
cp .env.example .env   # API 키 입력
jupyter lab
```
