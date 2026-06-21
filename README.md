# SteamLLM

SteamLLM은 Steam 게임의 현재 상태를 분석하고, 최근 뉴스/패치 정보와 리뷰 지표를 바탕으로 대화형 리포트를 생성하는 Streamlit 기반 게임 분석 에이전트입니다.

## 주요 기능

- Steam 게임명 추출 및 Steam Store 검색
- 현재 접속자 수, 전체/최근 리뷰 지표 수집
- Steam 뉴스 및 패치 노트 수집
- ChromaDB 기반 RAG 검색
- OpenAI LLM을 이용한 게임 상태 분석 리포트 생성
- 이전 대화와 마지막 분석 게임을 기억하는 채팅 UI

## 프로젝트 구조

```text
.
├── steam_api_project_13.py   # Streamlit 앱 및 분석 로직
├── requirements.txt          # Python 의존성 목록
├── chroma_db/                # ChromaDB 영속 저장소
└── .streamlit/
    └── secrets.toml          # OpenAI API 키 설정 파일
```

## 설치

Python 가상환경을 만든 뒤 의존성을 설치합니다.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux 환경에서는 가상환경 활성화 명령만 아래처럼 사용합니다.

```bash
source .venv/bin/activate
```

## 환경 변수 설정

`.streamlit/secrets.toml` 파일에 OpenAI API 키를 설정합니다.

```toml
OPENAI_API_KEY = "your-openai-api-key"
```

## 실행

```bash
streamlit run steam_api_project_13.py
```

실행 후 브라우저에서 Streamlit이 안내하는 로컬 주소로 접속하면 됩니다.

## 사용 예시

채팅 입력창에 아래와 같은 질문을 입력할 수 있습니다.

- `PUBG 지금 상태 어때?`
- `사이버펑크 2077 최근 패치 이후 평가는 어때?`
- `요즘 분석해볼 만한 스팀 게임 추천해줘`
- `방금 말한 게임 패치 내용 더 알려줘`

## 동작 방식

1. 사용자의 질문 의도를 `ANALYZE`, `DISCOVER`, `CHAT` 중 하나로 분류합니다.
2. 분석 대상 게임이 있으면 Steam Store에서 공식 게임명과 appid를 찾습니다.
3. Steam API와 Store API에서 접속자 수, 리뷰 통계, 뉴스 데이터를 수집합니다.
4. 수집한 텍스트를 ChromaDB에 저장하고, 후속 질문에서는 관련 문서를 검색해 답변합니다.
5. OpenAI 모델이 수집된 근거와 대화 맥락을 바탕으로 한국어 답변을 생성합니다.

## 참고 사항

- OpenAI API 키가 없으면 LLM 응답 및 임베딩 생성이 동작하지 않습니다.
- `chroma_db/`는 분석 데이터가 저장되는 로컬 벡터 DB 디렉터리입니다.
- Steam API 또는 Store API 응답 상태에 따라 일부 지표가 비어 있을 수 있습니다.
