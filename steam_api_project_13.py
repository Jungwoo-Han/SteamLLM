

import streamlit as st
import os
import requests
import json
import re
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import quote

# LangChain Imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

import chromadb
from chromadb.config import Settings


# ---------------------------------------------------------
# 1. Config (설정)
# ---------------------------------------------------------
class Config:
    def __init__(self):
        # 보안을 위해 환경 변수 사용을 권장하지만, 테스트를 위해 직접 입력 가능
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "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") 
        self.model_name = "gpt-4o-mini"
        self.embedding_model = "text-embedding-3-small"

# ---------------------------------------------------------
# 2. LLM Client (AI 호출)
# ---------------------------------------------------------
class LLMClient:
    def __init__(self, config: Config):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.model_name

    def ask(self, prompt: str, system_message: str = "You are a helpful assistant.") -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ LLM Error: {str(e)}"

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
    # (주의: 실제 Config, OpenAIEmbeddings, Document, RecursiveCharacterTextSplitter는 임포트가 되어 있어야 합니다)
    
    def __init__(self, config: Config, persist_dir="chroma_db"):
        self.config = config

        # ---- OpenAI Embedding ----
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key
        )

        # ---- ChromaDB 초기화 (Persist 설정) ----
        # Settings에 persist_directory가 지정되면 자동으로 디스크에 저장합니다.
        self.client = chromadb.Client(
            Settings(
                anonymized_telemetry=False,
                persist_directory=persist_dir
            )
        )

        # 컬렉션 생성 (없으면 새로 생성)
        self.collection = self.client.get_or_create_collection(
            name="steam_data",
            # embedding_function=self.embeddings (LangChain Embeddings 사용 시)
            metadata={"hnsw:space": "cosine"}
        )

        # ---- Text Splitter ----
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
# 7. Main Agent (컨트롤러)
# ---------------------------------------------------------
class SteamAdvisorAgent:
    def __init__(self):
        self.config = Config()
        self.llm = LLMClient(self.config)
        self.api = SteamAPIClient()
        self.rag = RAGManager(self.config)
        self.analyzer = GameHealthAnalyzer()
        self.extractor = GameNameExtractor(self.llm)

    # -----------------------------------------------------
    # 메인 실행 흐름
    # -----------------------------------------------------
    def run(self, user_input: str) -> str:
        print(f"\n🤖 분석 요청: '{user_input}'")

        # 1. 게임 식별
        game_info = self.extractor.extract_and_resolve(user_input)
        if not game_info:
            return "죄송합니다. 해당 게임을 스팀에서 찾을 수 없습니다."

        app_id = game_info["appid"]
        game_name = game_info["name"]
        print(f"🎯 대상 게임: {game_name} (ID: {app_id})")

        # -----------------------------------------------
        # 2. RAG에서 먼저 과거 데이터 검색 (이미 수집한 정보 활용)
        # -----------------------------------------------
        print("📚 기존 게임 데이터 검색 중...")
        rag_context = self.rag.search(
            query=user_input,
            appid=app_id,
            top_k=5
        )
        rag_text = self.rag.stringify_results(rag_context)

        # -----------------------------------------------
        # 3. 부족하면 API 호출로 최신 데이터 수집
        # -----------------------------------------------
        if not rag_context:
            print("⚠️ RAG 데이터 부족 → API 호출 시작")

        print("📡 실시간 Steam 데이터 수집 중...")
        players = self.api.get_current_players(app_id)
        reviews = self.api.get_review_stats(app_id)
        news = self.api.get_news(app_id, game_name)

        # 건강도 분석
        analysis = self.analyzer.analyze(players, reviews)

        # -----------------------------------------------
        # 4. 새로 수집한 데이터는 RAG DB에 저장
        # -----------------------------------------------
        print("💾 수집된 뉴스 데이터 RAG에 저장 중...")
        if news:
            self.rag.ingest(
                appid=app_id,
                game_name=game_name,
                texts=news,
                source="news"
            )

        # -----------------------------------------------
        # 5. 다시 RAG 검색 (fresh 데이터 포함)
        # -----------------------------------------------
        print("🔍 반영된 데이터 기반 RAG 검색 재실행...")
        updated_context = self.rag.search(
            query=f"{user_input} update patch bug",
            appid=app_id,
            top_k=5
        )
        evidence = self.rag.stringify_results(updated_context)

        # -----------------------------------------------
        # 6. LLM 리포트 생성
        # -----------------------------------------------
        print("✍️ 리포트 생성 중...")
        final_prompt = self._build_prompt(
            query=user_input,
            game_name=game_name,
            analysis=analysis,
            evidence=evidence,
            players=players
        )

        return self.llm.ask(final_prompt)

    # -----------------------------------------------------
    # 프롬프트 생성
    # -----------------------------------------------------
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



