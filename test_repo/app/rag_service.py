from openai import OpenAI
from app.config import Settings
from app.mariadb_vector_store import MariaDBVectorStore

class RagService:
    def __init__(self):
        self.vector_store = MariaDBVectorStore()
        #self.llm = OpenAI(api_key=Settings.OPENAI_API_KEY)
        self.llm = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

    def generate_answer(self, question:str):
        #1. 벡터검색
        docs = self.vector_store.similarity_search(question, k=5, filter=None)
#                                                     │      └> 유사도가 높은 우선순위 5개를 보여줘라
#                                                     └>사용자의 질문
        if not docs:
            return '검색된 문서가 없어서 RAG응답을 만들 수 없음'
        
        #2. 검색된 문서를 하나의 Context 텍스트로 병합
        context_text = '\n\n'.join([f'-[d.page_content]' for d in docs])

        #3. 모델에 전달할 프롬프트 구성
        prompt = f"""
당신은 검색 문서 기반으로 답변하는 RAG 전문가입니다.
답변은 반드시 한국어로 하세요.

[관련 문서들]
{context_text}

[사용자 질문]
{question}

위 문서를 근거로 정확하고 친절한 한국어 답변을 생성하세요.
"""

        #4. LLM 호출
        completion = self.llm.chat.completions.create(
            #model='gpt-40-mini',
            model="llama3",
            messages=[
                {"role":"system", "content":"당신은 문서 기반 RAG QA전문가 입니다."},
                {"role":"system", "content":"답변은 반드시 한국어로만 하세요."},
                {"role":"user", "content":prompt}
            ]
        )

        return completion.choices[0].message.content