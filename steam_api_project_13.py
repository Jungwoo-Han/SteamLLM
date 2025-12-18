import streamlit as st
import os
import requests
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import quote

# LangChain Imports (최신 규격)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb

# ---------------------------------------------------------
# 1. Config (설정)
# ---------------------------------------------------------
class Config:
    def __init__(self):
        self.openai_api_key = st.secrets.get("OPENAI_API_KEY")
        # 최신 gpt-5.1 모델 설정
        self.model_name = "gpt-5.1" 
        self.embedding_model = "text-embedding-3-large"

# ---------------------------------------------------------
# 2. LLM Client (LangChain .invoke 방식 적용)
# ---------------------------------------------------------
class LLMClient:
    def __init__(self, config: Config):
        # langchain-openai 0.3.x 규격 적용
        self.llm = ChatOpenAI(
            api_key=config.openai_api_key,
            model=config.model_name,
            reasoning={
                "effort": "none",    # gpt-5.1의 추론 과정을 비활성화하여 속도 최적화
                "summary": "auto",
            },
            verbosity="high",        # 디버깅용 로그 활성화
            temperature=0.2
        )

    def ask(self, prompt: str, system_message: str = "You are a helpful assistant.") -> str:
        try:
            # 최신 메시지 객체 생성 및 .invoke() 호출
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"❌ LLM Error: {str(e)}"

# ---------------------------------------------------------
# Intent Classifier (사용자 의도 분류)
# ---------------------------------------------------------
class IntentClassifier:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def classify(self, user_input: str) -> str:
        prompt = f"""
[Role] Steam Interaction Router
분석 방향을 결정하기 위해 사용자 입력을 다음 중 하나로 분류하세요:

1. ANALYZE: 특정 게임 하나를 지칭하여 상태, 정보, 패치 등을 묻는 경우. (예: "배그 요즘 어때?", "사펑 할만함?")
2. DISCOVER: 무엇을 분석할지 고민 중이거나, 요즘 트렌디한 게임 리스트를 보고 싶어 하는 경우. (예: "요즘 분석해볼 만한 게임 있어?", "스팀 인기작 추천해줘")
3. CHAT: 게임 분석과 무관한 인사, 일상 대화, 혹은 서비스 사용법 질문. (예: "안녕?", "로그라이크가 뭐야?", "너는 누구니?")

반드시 단어 하나(ANALYZE, DISCOVER, CHAT)만 출력하세요.
입력: "{user_input}" """
        return self.llm.ask(prompt).strip().upper()

# ---------------------------------------------------------
# 3. Steam API Client (데이터 수집)
# ---------------------------------------------------------
class SteamAPIClient:
    def __init__(self):
        self.base_url = "http://api.steampowered.com"
        self.store_url = "https://store.steampowered.com"
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def _clean_html_text(self, raw_html: str) -> str:
        if not raw_html: return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        text = soup.get_text(separator=" ")
        text = re.sub(r'http\S+', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def get_news(self, app_id: int, game_name: str, count: int = 5) -> list[str]:
        url = f"{self.base_url}/ISteamNews/GetNewsForApp/v0002/"
        params = {'appid': app_id, 'count': count, 'maxlength': 0, 'format': 'json'}
        try:
            response = requests.get(url, params=params)
            data = response.json()
            news_items = data.get('appnews', {}).get('newsitems', [])
            processed = []
            for item in news_items:
                date_str = datetime.fromtimestamp(item['date']).strftime('%Y-%m-%d')
                content = self._clean_html_text(item['contents'])
                processed.append(f"게임: {game_name}\n제목: {item['title']}\n날짜: {date_str}\n내용: {content}\n")
            return processed
        except Exception:
            return []

    def get_current_players(self, app_id: int) -> int:
        url = f"{self.base_url}/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
        try:
            resp = requests.get(url, params={'appid': app_id})
            return resp.json().get('response', {}).get('player_count', 0)
        except:
            return 0

    def get_review_stats(self, app_id: int) -> dict:
        """스팀 상점 페이지 리뷰 데이터 파싱"""
        url = f"{self.store_url}/appreviews/{app_id}"
        try:
            # 전체 리뷰
            res_all = requests.get(url, params={'json': 1, 'language': 'all', 'num_per_page': 0}, headers=self.headers).json()
            summary = res_all.get('query_summary', {})
            total_pos = summary.get('total_positive', 0)
            total_count = summary.get('total_reviews', 1) # div 0 방지
            
            # 최근 리뷰 (30일)
            res_recent = requests.get(url, params={'json': 1, 'language': 'all', 'filter': 'recent', 'num_per_page': 100}, headers=self.headers).json()
            recent_reviews = res_recent.get('reviews', [])
            
            recent_count = len(recent_reviews)
            recent_pos_count = sum(1 for r in recent_reviews if r.get('voted_up'))
            
            all_percent = (total_pos / total_count) * 100
            recent_percent = (recent_pos_count / recent_count * 100) if recent_count > 0 else all_percent

            return {
                "recent_percent": round(recent_percent, 1),
                "all_percent": round(all_percent, 1),
                "sample_count": recent_count
            }
        except:
            return {"recent_percent": 0, "all_percent": 0, "sample_count": 0}

# ---------------------------------------------------------
# 4. RAG Manager (벡터 검색)
# ---------------------------------------------------------



class RAGManager:
    def __init__(self, config: Config, persist_dir="chroma_db"):
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key
        )

        # 최신 버전의 방식인 PersistentClient 사용
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name="steam_data",
            metadata={"hnsw:space": "cosine"}
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    # ---------------------------------------------------------
    # ① 문서 삽입 (게임별 뉴스/리포트/데이터 저장)
    # ---------------------------------------------------------
    def ingest(self, appid: int, game_name: str, texts: list[str], source: str):
        """ 
        appid: 게임 Steam ID 
        game_name: Official title
        texts: 저장할 텍스트 
        source: 뉴스/리포트/리뷰 등 종류 ("news", "report")
        """
        if not texts:
            return

        docs = [Document(page_content=t, metadata={"appid": appid, "game": game_name, "source": source}) 
                for t in texts]

        splits = self.text_splitter.split_documents(docs)

        documents_to_add = []
        embeddings_to_add = []
        metadatas_to_add = []
        ids_to_add = []

        for idx, d in enumerate(splits):
            doc_id = f"{appid}-{source}-{idx}-{hash(d.page_content)}"
            
            documents_to_add.append(d.page_content)
            embeddings_to_add.append(self.embeddings.embed_query(d.page_content))
            metadatas_to_add.append(d.metadata)
            ids_to_add.append(doc_id)

        # 일괄 add로 성능 개선 (선택적)
        if documents_to_add:
            self.collection.add(
                documents=documents_to_add,
                embeddings=embeddings_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )

    # ---------------------------------------------------------
    # ② 쿼리 + 필터링 검색
    # ---------------------------------------------------------
    def search(self, query: str, appid: int = None, top_k: int = 5) -> list[dict]:
        """
        query: 자연어 질의
        appid: 특정 게임만 검색하려면 지정
        return: 문서 리스트
        """

        query_embedding = self.embeddings.embed_query(query)

        # 메타데이터 필터 적용 (appid가 있으면 해당 게임 자료만 검색)
        where = {"appid": appid} if appid else {}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )

        if not results["documents"]:
            return []

        # ChromaDB 결과를 LangChain Document 형태로 변환
        docs = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append({"content": doc, "metadata": meta})

        return docs

    # ---------------------------------------------------------
    # ③ 검색 결과 문자열로 정제 (리포트 작성용)
    # ---------------------------------------------------------
    def stringify_results(self, docs: list[dict]) -> str:
        if not docs:
            return "관련 문서 없음"

        formatted = []
        for d in docs:
            meta = d["metadata"]
            src = meta.get("source", "unknown")
            formatted.append(f"[{src}] {d['content']}")

        return "\n\n".join(formatted)

# ---------------------------------------------------------
# 5. Game Name Extractor (게임명 추출)
# ---------------------------------------------------------
class GameNameExtractor:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def _search_steam_store(self, query: str) -> dict:
        """스팀 상점 검색 API 활용"""
        try:
            url = f"https://store.steampowered.com/api/storesearch/?term={quote(query)}&cc=us"
            res = requests.get(url, timeout=3).json()
            items = res.get('items', [])
            if items:
                return {"appid": items[0]['id'], "name": items[0]['name']} # 첫 번째 결과 반환
        except:
            pass
        return None

    def extract_and_resolve(self, user_input: str) -> dict:
        """사용자 입력 -> LLM 추출 -> 스팀 ID 검색"""
        # 1. LLM에게 게임 이름만 뽑아달라고 요청
        prompt = f"""
You are an expert Steam Store search assistant.
Your task is to identify the game mentioned in the user's input and translate it into its **Official English Steam Store Title**.

Rules:
1. Identify the game name from the Korean input.
2. Convert Korean abbreviations or nicknames into the full official English title.
3. Return **ONLY** the official title string. Do not output any other text or punctuation.

Examples:
- Input: "배그 요즘 어때?" -> Output: PUBG: BATTLEGROUNDS
- Input: "배틀그라운드 복귀할까?" -> Output: PUBG: BATTLEGROUNDS
- Input: "사펑 버그 고쳐짐?" -> Output: Cyberpunk 2077
- Input: "스듀 멀티 돼?" -> Output: Stardew Valley
- Input: "레데리2 할인해?" -> Output: Red Dead Redemption 2
- Input: "롤 같은 게임 추천해줘" -> Output: League of Legends

User Input: "{user_input}"
"""
        game_name_candidate = self.llm.ask(prompt).strip().replace('"', '')
        
        # 2. 스팀 API로 ID 찾기
        result = self._search_steam_store(game_name_candidate)
        if result:
            return result
        
        # 검색 실패 시, 입력된 텍스트 그대로 다시 시도
        return self._search_steam_store(user_input)

# ---------------------------------------------------------
# 6. Health Analyzer (분석 로직)
# ---------------------------------------------------------
class GameHealthAnalyzer:
    def analyze(self, players: int, review_stats: dict) -> dict:
        recent = review_stats['recent_percent']
        overall = review_stats['all_percent']
        
        return {
            "status": "RISING" if (recent - overall) > 10 else "STAGNANT",
            "warning": "LOW_POPULATION" if players < 1000 else "ACTIVE",
            "recent_score": recent,
            "all_score": overall
        }


# ---------------------------------------------------------
# 7. Main Agent (컨트롤러 - 리팩토링 버전)
# ---------------------------------------------------------
class SteamAdvisorAgent:
    def __init__(self):
        self.config = Config()
        self.llm = LLMClient(self.config)
        self.api = SteamAPIClient()
        self.rag = RAGManager(self.config)
        self.analyzer = GameHealthAnalyzer()
        self.extractor = GameNameExtractor(self.llm)
        self.classifier = IntentClassifier(self.llm) # 신규 추가된 부품

    # -----------------------------------------------------
    # Step 0. 메인 실행 흐름 (라우터)
    # -----------------------------------------------------
    def run(self, user_input: str) -> str:
        """사용자의 의도를 분류하고 적절한 핸들러로 라우팅합니다."""
        intent = self.classifier.classify(user_input)
        print(f"🔍 의도 분류 결과: {intent}")

        if intent == "ANALYZE":
            return self._handle_analysis(user_input)
        elif intent == "DISCOVER":
            return self._handle_discovery(user_input)
        else:
            return self._handle_chat(user_input)

    # -----------------------------------------------------
    # 브랜치 1. 분석 핸들러 (1-1 vs 1-2 분기)
    # -----------------------------------------------------
    def _handle_analysis(self, user_input: str) -> str:
        """특정 게임 분석을 처리하며, RAG 데이터 존재 여부에 따라 분기합니다."""
        game_info = self.extractor.extract_and_resolve(user_input)
        if not game_info:
            return "죄송합니다. 해당 게임을 스팀에서 찾을 수 없습니다. 정확한 게임명을 입력해 주세요."

        app_id = game_info["appid"]
        game_name = game_info["name"]
        print(f"🎯 대상 게임: {game_name} (ID: {app_id})")

        # RAG 데이터 존재 여부 확인
        existing_docs = self.rag.search(query=user_input, appid=app_id, top_k=1)

        if not existing_docs:
            # [Branch 1-1] 데이터 없음: 정형 리포트 파이프라인 실행
            print(f"🆕 '{game_name}' 신규 데이터 수집 모드 실행")
            return self._run_full_report_pipeline(app_id, game_name, user_input)
        else:
            # [Branch 1-2] 데이터 있음: 비정형 대화 모드 실행
            print(f"💬 '{game_name}' 기존 데이터 기반 대화 모드 실행")
            context = self.rag.search(query=user_input, appid=app_id, top_k=5)
            evidence = self.rag.stringify_results(context)
            return self._run_conversational_qa(game_name, user_input, evidence)

    # -----------------------------------------------------
    # 브랜치 1-1. 정형 리포트 파이프라인 (기존 run 로직)
    # -----------------------------------------------------
    def _run_full_report_pipeline(self, app_id: int, game_name: str, user_input: str) -> str:
        """실시간 데이터를 수집하고 정형화된 리포트를 생성합니다."""
        print("📡 실시간 Steam 데이터 수집 중...")
        players = self.api.get_current_players(app_id)
        reviews = self.api.get_review_stats(app_id)
        news = self.api.get_news(app_id, game_name)

        # 건강도 분석
        analysis = self.analyzer.analyze(players, reviews)

        # 데이터 RAG 저장
        if news:
            print("💾 수집된 뉴스 데이터 RAG에 저장 중...")
            self.rag.ingest(appid=app_id, game_name=game_name, texts=news, source="news")

        # 증거 데이터 검색
        updated_context = self.rag.search(query=f"{user_input} update patch bug", appid=app_id)
        evidence = self.rag.stringify_results(updated_context)

        print("✍️ 정형 리포트 생성 중...")
        final_prompt = self._build_prompt(user_input, game_name, analysis, evidence, players)
        return self.llm.ask(final_prompt)

    # -----------------------------------------------------
    # 브랜치 1-2. 비정형 대화 (QA)
    # -----------------------------------------------------
    def _run_conversational_qa(self, game_name: str, query: str, evidence: str) -> str:
        """기존 지식을 바탕으로 자연스러운 대화를 나눕니다."""
        prompt = f"""
당신은 스팀 게임 전문 분석가입니다. 아래 제공된 [과거 분석 데이터]를 바탕으로 '{game_name}'에 대한 사용자의 질문에 답하세요.
리포트 형식을 따르지 말고, 질문에 직접적이고 친절하게 대화하듯 답변하십시오.

[과거 분석 데이터]
{evidence}

사용자 질문: "{query}"
        """
        return self.llm.ask(prompt)

    # -----------------------------------------------------
    # 브랜치 2. 분석 가이드 (Discovery)
    # -----------------------------------------------------
    def _handle_discovery(self, user_input: str) -> str:
        prompt = f"""
사용자가 분석할 만한 스팀 게임 리스트를 찾고 있습니다. 
최근 트렌드, 대규모 업데이트, 혹은 갑작스러운 인기 상승 중인 게임 5가지를 추천하고 그 이유를 설명하세요.
마지막에는 리스트 중 궁금한 게임의 이름을 입력하면 상세 분석을 시작하겠다는 안내를 포함하세요.

사용자 질문: "{user_input}"
        """
        return self.llm.ask(prompt)

    # -----------------------------------------------------
    # 브랜치 3. 일반 대화 (Chat)
    # -----------------------------------------------------
    def _handle_chat(self, user_input: str) -> str:
        prompt = f"당신은 친절한 AI 게임 파트너입니다. 게임에 관한 일반적인 상식이나 일상적인 대화에 답해 주세요. 질문: {user_input}"
        return self.llm.ask(prompt)

    # (기존 _build_prompt 메서드는 그대로 유지)
    def _build_prompt(self, query, game_name, analysis, evidence, players):
        return f"""
[Role] Steam Analyst Agent
[Task] Analyze '{game_name}' based on data and answer the user query: "{query}"

[Game Data Summary]
- 현재 동접자 수: {players:,}명
- 최근 평가: {analysis['recent_score']}%
- 전체 평가: {analysis['all_score']}%
- 상태: {analysis['status']} (경고: {analysis['warning']})

[News & Update Evidence from RAG]
{evidence}

[Output Format]
한국어 리포트를 작성하라.
1. 게임 상태 요약 (성장 / 안정 / 하락)
2. 최근 업데이트·패치 내용을 Fact 기반으로 설명 (RAG evidence 사용)
3. 유저 질문 의도에 맞춘 최종 추천 결론 제시
        """




# ---------------------------------------------------------
# 🧪 Streamlit UI 코드 시작
# ---------------------------------------------------------

# @st.cache_resource를 사용하여 Agent 인스턴스는 한 번만 생성되도록 최적화
@st.cache_resource
def load_agent():
    return SteamAdvisorAgent()

if __name__ == "__main__":
    agent = load_agent()

    st.set_page_config(page_title="Steam Health Advisor", layout="centered")
    st.title("Steam 게임 건전성 분석 에이전트 🎮")

    user_query = st.text_input("어떤 게임에 대해 알고 싶으신가요?")

    if st.button("분석 실행", type="primary"):
        if user_query:
            with st.spinner('Steam 데이터 수집 및 AI 분석 중...'):
                report_markdown = agent.run(user_query)
                
                st.success("분석 완료!")
                st.markdown(report_markdown) 
        else:
            st.error("질문을 입력해 주세요.")



