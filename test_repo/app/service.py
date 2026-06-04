import json
import pymysql
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from app.config import Settings
from app.mariadb_vector_store import MariaDBVectorStore
from app.sample_docs import sample_documents
import math

class vectorService:
    def __init__(self):
        self.store = MariaDBVectorStore()

    # ---------------------------------------------------
    # 1) 샘플 문서 저장 (Spring의 loadSampleDocuments() 대응)
    # ---------------------------------------------------
    def load_sample_docs(self):
        docs = [
            Document(page_content=d["text"], metadata=d["metadata"])
            for d in sample_documents
        ]

        self.store.add_documents(docs)
        return len(docs)


    # ---------------------------------------------------
    # 2) search1 — 기본 similaritySearch
    # ---------------------------------------------------
    def search1(self, query: str):
        return self.store.similarity_search(query, k=5)


    # ---------------------------------------------------
    # 3) search2 — threshold + filterExpression
    # "author = john" AND "article_type = blog"
    # ---------------------------------------------------
    def search2(self, query: str):
        filter_ = {"author": "john", "article_type": "blog"}
        return self.store.similarity_search(query, k=TOP_K, filter=filter_)


    # ---------------------------------------------------
    # 4) Score 출력
    # similarity_search_with_score() 호출
    # ---------------------------------------------------
    def search_with_scores(self, query: str):
        return self.store.similarity_search_with_score(query, k=2)


    # ---------------------------------------------------
    # 5) Native SQL 조회 (Spring의 JdbcTemplate 기능)
    # ---------------------------------------------------
    def native_query(self):
        conn = pymysql.connect(
            host=Settings.DB_HOST,
            port=Settings.DB_PORT,
            user=Settings.DB_USER,
            password=Settings.DB_PASS,
            database=Settings.DB_NAME,
            cursorclass=pymysql.cursors.DictCursor
        )
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {Settings.DB_TABLE} LIMIT 20")
        rows = cur.fetchall()
        conn.close()
        return rows