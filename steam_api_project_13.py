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
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import chromadb
from chromadb.config import Settings
import streamlit as st


# ---------------------------------------------------------
# 1. Config (설정)
# ---------------------------------------------------------
class Config:
    def __init__(self):
        # st.secrets에서 키를 가져오고, 없으면 None을 반환
        self.openai_api_key = st.secrets.get("OPENAI_API_KEY")
        self.model_name = "gpt-4o-mini"
        self.embedding_model = "text-embedding-3-large"

        
# ---------------------------------------------------------
# LLM Client (역사 참조 기능 추가)
# ---------------------------------------------------------
class LLMClient:
    def __init__(self, config):
        # gpt-4o 모델 사용, 답변의 일관성을 위해 temperature는 낮게 설정
        self.llm = ChatOpenAI(
            api_key=config.openai_api_key,
            model="gpt-4o", 
            temperature=0.1
        )

    def ask(self, prompt: str, history: list = None, system_message: str = "You are a helpful Steam game assistant.") -> str:
        try:
            messages = [SystemMessage(content=system_message)]
            
            # 이전 대화 기록이 있다면 메시지 객체로 변환하여 추가 (최근 5개 권장)
            if history:
                for msg in history[-5:]:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
            
            # 마지막 현재 질문 추가
            messages.append(HumanMessage(content=prompt))
            
            response = self.llm.invoke(messages)
            return response.content if response.content else ""
        except Exception as e:
            return f"ERROR: {str(e)}"
        
# ---------------------------------------------------------
# 2. Intent Classifier (의도 분류)
# ---------------------------------------------------------


class IntentClassifier:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def classify(self, user_input: str, history: list = None) -> str:
        system_prompt = """
[Role] Steam Interaction Router
이전 대화 맥락을 고려하여 사용자 질문의 의도를 분류하세요. 분석 방향을 결정하기 위해 사용자 입력을 다음 중 하나로 분류하세요:
1. ANALYZE: 특정 게임 하나를 지칭하여 상태, 정보, 패치 등을 묻거나 이전 게임에 대한 추가 정보를 요청하는 경우. (예: "배그 요즘 어때?", "사펑 할만함?", "더 자세히 알려줘", "아까 말한 패치 내용은?")
2. DISCOVER: 무엇을 분석할지 고민 중이거나, 요즘 트렌디한 게임 리스트를 보고 싶어 하는 경우. (예: "요즘 분석해볼 만한 게임 있어?", "스팀 인기작 추천해줘")
3. CHAT: 게임 분석과 무관한 인사, 일상 대화, 혹은 서비스 사용법 질문. (예: "안녕?", "로그라이크가 뭐야?", "너는 누구니?")

질문: "{user_input}"
결과(단어 하나만):"""
        return self.llm.ask(user_input, history=history, system_message=system_prompt).strip().upper()

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
# Game Name Extractor (맥락 기반 추출)
# ---------------------------------------------------------
class GameNameExtractor:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract_and_resolve(self, user_input: str, history: list = None, last_game_info: dict = None) -> dict:
        prompt = f"""
You are an expert Steam Store search assistant.
Your task is to identify the game mentioned in the user's input and translate it into its **Official English Steam Store Title**.

[Rules]
1. Identify the game name from the Korean input.
2. Convert Korean abbreviations or nicknames (e.g., '사펑', '배그') into full official English titles.
3. Return **ONLY** the official title string. 
4. If the user refers to a previously mentioned game (e.g., "그 게임", "이거", "패치 내용"), return 'SAME'.
5. If no game is mentioned at all, return 'NONE'.

[Examples]
- "배그 동접자 어때?" -> PUBG: BATTLEGROUNDS
- "사펑 할만해?" -> Cyberpunk 2077
- "그 게임 패치 내역은?" -> SAME
- "아까 말한 거 정보 더 줘" -> SAME

User Input: "{user_input}"
결과:"""
        
        # LLM 호출 및 결과 정제
        extracted = self.llm.ask(prompt, history=history).strip().replace('"', '')
        
        # LLM이 간혹 "결과: PUBG" 식으로 출력하는 경우를 대비해 불필요한 태그 제거
        extracted = extracted.split(':')[-1].strip()

        # 1. 이전 게임 유지 조건
        if "SAME" in extracted.upper() or extracted.upper() == "NONE" or len(extracted) < 2:
            print(f"🔄 [Extractor] 기존 게임 문맥 유지: {last_game_info.get('name') if last_game_info else 'None'}")
            return last_game_info

        # 2. 새로운 게임 검색
        search_url = f"https://store.steampowered.com/api/storesearch/?term={quote(extracted)}&cc=us"
        try:
            res = requests.get(search_url, timeout=5).json()
            if res.get('items'):
                found_game = {
                    "appid": res['items'][0]['id'],
                    "name": res['items'][0]['name']
                }
                print(f"🎯 [Extractor] 신규 게임 탐지: {found_game['name']}")
                return found_game
        except Exception as e:
            print(f"⚠️ [Extractor] 검색 중 오류 발생: {e}")
        
        # 검색 실패 시 마지막 게임 정보로 폴백
        return last_game_info

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
# 7. Main Agent (gpt-4o Context-Aware Version)
# ---------------------------------------------------------
class SteamAdvisorAgent:
    def __init__(self):
        self.config = Config()
        self.llm = LLMClient(self.config)
        self.api = SteamAPIClient()
        self.rag = RAGManager(self.config)
        self.analyzer = GameHealthAnalyzer()
        self.extractor = GameNameExtractor(self.llm)
        self.classifier = IntentClassifier(self.llm)

    def run(self, user_input: str, history: list, last_game_info: dict) -> tuple:
        """
        사용자 입력과 대화 기록, 이전 게임 정보를 받아 최종 응답을 생성합니다.
        """
        # [Step 1] 문맥 기반 의도 분류
        intent = self.classifier.classify(user_input, history)
        print(f"🔍 [System] Detected Intent: {intent}")

        # [Step 2] 의도에 따른 분기 처리
        if intent == "ANALYZE":
            return self._handle_analysis(user_input, history, last_game_info)
        elif intent == "DISCOVER":
            return self._handle_discovery(user_input, history), last_game_info
        else:
            return self._handle_chat(user_input, history), last_game_info

    # -----------------------------------------------------
    # 브랜치 1. 분석 핸들러 (맥락 기반 데이터 관리)
    # -----------------------------------------------------
    def _handle_analysis(self, user_input: str, history: list, last_game_info: dict) -> tuple:
        # 1. 게임 정보 식별 (지칭어 해결 포함)
        game_info = self.extractor.extract_and_resolve(user_input, history, last_game_info)
        
        if not game_info:
            error_msg = "🤔 분석할 게임을 정확히 찾지 못했습니다. 게임 제목을 다시 한번 말씀해 주시겠어요?"
            return error_msg, last_game_info

        app_id = game_info["appid"]
        game_name = game_info["name"]

        # 2. RAG 데이터 존재 여부 확인 (기분석 여부 체크)
        # top_k=1로 검색하여 이 게임에 대한 학습 데이터가 한 개라도 있는지 확인합니다.
        existing_docs = self.rag.search(query=user_input, appid=app_id, top_k=1)

        if not existing_docs:
            # [Branch 1-1] 신규 리포트 생성 (Data Collection + Ingestion)
            print(f"🆕 [System] New Analysis for: {game_name}")
            response = self._run_full_report_pipeline(app_id, game_name, user_input, history)
        else:
            # [Branch 1-2] 기존 데이터 기반 연속 QA (Conversational RAG)
            print(f"💬 [System] Continuing Conversation for: {game_name}")
            context_docs = self.rag.search(query=user_input, appid=app_id, top_k=5)
            evidence = self.rag.stringify_results(context_docs)
            response = self._run_conversational_qa(game_name, user_input, evidence, history)
            
        return response, game_info

    # -----------------------------------------------------
    # 브랜치 1-1. 정밀 리포트 파이프라인 (GPT-4o 전용 프롬프트)
    # -----------------------------------------------------
    def _run_full_report_pipeline(self, app_id: int, game_name: str, user_input: str, history: list) -> str:
        # 데이터 수집 및 RAG 저장
        players = self.api.get_current_players(app_id)
        reviews = self.api.get_review_stats(app_id)
        news_list = self.api.get_news(app_id, game_name)

        if news_list:
            self.rag.ingest(appid=app_id, game_name=game_name, texts=news_list, source="news")

        analysis = self.analyzer.analyze(players, reviews)
        evidence = self.rag.stringify_results(self.rag.search(query=user_input, appid=app_id))

        prompt = f"""
[Role] Steam Game Strategic Analyst
[Context] 사용자가 '{game_name}'에 대한 정밀 분석을 처음 요청했습니다. 실시간 API 데이터를 바탕으로 전문적인 리포트를 작성하세요.

[Steam Live Metrics]
- 실시간 동시 접속자: {players:,}명 (전 세계 기준)
- 최근 유저 긍정 응답률: {analysis['recent_score']}%
- 전체 누적 평점: {analysis['all_score']}%
- 현재 게임 건강도: {analysis['status']} (비고: {analysis['warning']})

[Technical News & Patch Notes (RAG)]
{evidence}

[Requirements]
1. **상태 진단**: 현재 게임의 활성도와 유저 민심을 날카롭게 요약하십시오.
2. **패치 하이라이트**: 최근 뉴스 데이터 중 유저가 반드시 알아야 할 패치나 이슈를 Fact 중심으로 정리하십시오.
3. **투자 및 플레이 제언**: 유저의 질문("{user_input}")을 고려하여, 이 게임에 지금 시간이나 비용을 투자할 가치가 있는지 최종 결론을 내리십시오.

답변은 한국어로 작성하며, 전문적이면서도 가독성 좋게(Markdown 활용) 구성하세요.
"""
        return self.llm.ask(prompt, history=history)

    # -----------------------------------------------------
    # 브랜치 1-2. 심층 대화 (Follow-up QA)
    # -----------------------------------------------------
    def _run_conversational_qa(self, game_name: str, query: str, evidence: str, history: list) -> str:
        prompt = f"""
[Role] Steam Intelligence Advisor
[Context] 당신은 이미 '{game_name}'에 대한 정밀 분석을 마친 상태입니다. 
당신이 이미 알고 있는 아래 [지식 근거]를 바탕으로 사용자의 추가 질문에 답하세요.

[지식 근거]
{evidence}

[Constraints]
- 이미 리포트를 작성했으므로, 리포트 형식을 반복하지 마십시오.
- 친절한 전문가 파트너로서 자연스럽게 대화하며 질문("{query}")에 직접적인 정보를 제공하십시오.
- 이전 대화 내용(History)을 참조하여 문맥에 어긋나지 않게 답변하십시오.
"""
        return self.llm.ask(prompt, history=history)

    # -----------------------------------------------------
    # 브랜치 2 & 3. Discovery & Chat
    # -----------------------------------------------------
    def _handle_discovery(self, user_input: str, history: list) -> str:
        prompt = f"""
사용자가 분석해볼 만한 스팀 게임을 찾고 있습니다. 
사용자의 질문: "{user_input}"

다음 원칙에 따라 답변하세요:
1. 사용자의 관심사에 맞는 5~7개의 스팀 게임 리스트를 제안하세요.
2. 각 게임이 왜 '분석해볼 가치가 있는지'(최근 패치, 동접자 급증, 논란 등) 짧게 설명하세요.
3. "이 중 궁금한 게임의 이름을 입력하시면 상세 분석 리포트를 작성해 드립니다"라는 안내를 포함하세요.
        """
        return self.llm.ask(prompt, history=history)

    def _handle_chat(self, user_input: str, history: list) -> str:
        prompt = f"""
게임 전문 지식을 갖춘 친절한 AI 파트너로서 사용자("{user_input}")와 일상적인 대화를 나누세요."""
        return self.llm.ask(prompt, history=history)




# ---------------------------------------------------------
# 🧪 Streamlit UI 코드 (대화 기록 유지 버전)
# ---------------------------------------------------------

@st.cache_resource
def load_agent():
    return SteamAdvisorAgent()

if __name__ == "__main__":
    agent = load_agent()
    st.set_page_config(page_title="Steam Health Advisor", layout="wide")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_game" not in st.session_state:
        st.session_state.last_game = None # 마지막 대화 게임 정보 저장

    st.title("Steam 게임 분석 에이전트 🎮")

    # 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("질문을 입력하세요 (예:  요즘 분석해볼 만한 게임 있어? -> 호그와트 레거시 어때? -> 패치 내역은?)"):
        
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner('문맥 파악 및 분석 중...'):
                # 에이전트 실행 시 역사(messages)와 마지막 게임(last_game) 전달
                response, updated_game = agent.run(
                    user_query, 
                    st.session_state.messages[:-1], # 현재 질문 제외한 기록
                    st.session_state.last_game
                )
                
                st.markdown(response)
                
                # 세션 업데이트
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.last_game = updated_game # 마지막 게임 정보 갱신

    # 사이드바 정보
    if st.session_state.last_game:
        st.sidebar.info(f"📍 현재 분석 대상: {st.session_state.last_game['name']}")
    
    if st.sidebar.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.last_game = None
        st.rerun()