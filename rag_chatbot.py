import os
import re
import json
from collections import defaultdict
from pathlib import Path

import chromadb
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer


# =========================================================
# 1. 기본 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "support_policy"

GEMINI_MODEL = "gemini-3.6-flash"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

VECTOR_SEARCH_COUNT = 80
FINAL_CONTEXT_COUNT = 14


# =========================================================
# 2. Streamlit 설정
# =========================================================

st.set_page_config(
    page_title="강원 소상공인 혜택 도우미",
    page_icon="💼",
    layout="centered",
)


# =========================================================
# 화면 디자인
# GANGWON_SUPPORT_THEME_V1
# =========================================================

st.markdown(
    """
    <style>

    /* ====================================================
       전체 앱 배경
    ==================================================== */

    .stApp {
        background:
            linear-gradient(
                180deg,
                #F5F8FA 0%,
                #F8FAFB 100%
            );
    }


    /* ====================================================
       메인 콘텐츠 폭
    ==================================================== */

    .block-container {
        max-width: 780px;
        padding-top: 2.7rem;
        padding-bottom: 4rem;
    }


    /* ====================================================
       전체 기본 글자
    ==================================================== */

    html,
    body,
    [class*="css"] {
        font-family:
            "Pretendard",
            "Noto Sans KR",
            "Apple SD Gothic Neo",
            sans-serif;
    }


    p {
        line-height: 1.72;
        color: #334155;
    }


    /* ====================================================
       제목
    ==================================================== */

    h1 {
        color: #17324D !important;
        font-size: 2.55rem !important;
        font-weight: 800 !important;
        line-height: 1.22 !important;
        letter-spacing: -0.045em !important;
        margin-bottom: 0.65rem !important;
    }


    h2 {
        color: #17324D !important;
        font-size: 1.85rem !important;
        font-weight: 750 !important;
        line-height: 1.3 !important;
        letter-spacing: -0.035em !important;
    }


    h3 {
        color: #203B55 !important;
        font-size: 1.28rem !important;
        font-weight: 720 !important;
        letter-spacing: -0.025em !important;
        margin-top: 1.35rem !important;
        margin-bottom: 0.65rem !important;
    }


    /* ====================================================
       작은 안내 문구
    ==================================================== */

    [data-testid="stCaptionContainer"] {
        color: #7A8795 !important;
        font-size: 0.9rem !important;
        line-height: 1.55 !important;
    }


    /* ====================================================
       라디오 영역
    ==================================================== */

    [data-testid="stRadio"] {
        background: #FFFFFF;
        padding: 1rem 1.1rem;
        border-radius: 16px;
        border: 1px solid #E3EAF0;
        box-shadow:
            0 6px 18px rgba(31, 50, 70, 0.045);
        margin-top: 0.35rem;
        margin-bottom: 1.5rem;
    }


    [data-testid="stRadio"] label {
        color: #334155 !important;
        font-weight: 600 !important;
    }


    /* ====================================================
       텍스트 입력창
    ==================================================== */

    .stTextArea textarea {
        background: #FFFFFF !important;
        color: #24364B !important;
        border: 1px solid #DCE5EC !important;
        border-radius: 15px !important;
        min-height: 118px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        box-shadow:
            0 5px 16px rgba(30, 52, 72, 0.035);
    }


    .stTextArea textarea:focus {
        border-color: #2E8C8C !important;
        box-shadow:
            0 0 0 3px rgba(46, 140, 140, 0.10)
            !important;
    }


    .stTextArea textarea::placeholder {
        color: #9AA6B2 !important;
    }


    /* ====================================================
       버튼
    ==================================================== */

    .stButton > button {
        border-radius: 12px !important;
        min-height: 46px !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        letter-spacing: -0.01em !important;
        border: 1px solid #D8E2E9 !important;
        background: #FFFFFF !important;
        color: #29435B !important;
        transition:
            all 0.18s ease !important;
    }


    .stButton > button:hover {
        border-color: #2E8C8C !important;
        color: #176A6A !important;
        background: #F4FBFA !important;
        transform: translateY(-1px);
        box-shadow:
            0 5px 14px rgba(46, 140, 140, 0.10);
    }


    .stButton > button:focus {
        box-shadow:
            0 0 0 3px rgba(46, 140, 140, 0.10)
            !important;
    }


    /* ====================================================
       Primary 버튼
    ==================================================== */

    .stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background:
            linear-gradient(
                135deg,
                #17324D,
                #24536C
            ) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow:
            0 7px 18px rgba(23, 50, 77, 0.17);
    }


    .stButton > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background:
            linear-gradient(
                135deg,
                #21445E,
                #2D6677
            ) !important;
        color: #FFFFFF !important;
    }


    /* ====================================================
       Divider
    ==================================================== */

    hr {
        border: none !important;
        border-top: 1px solid #E2E8EE !important;
        margin-top: 1.65rem !important;
        margin-bottom: 1.65rem !important;
    }


    /* ====================================================
       Progress
    ==================================================== */

    [data-testid="stProgressBar"] {
        margin-top: 0.4rem;
        margin-bottom: 1.3rem;
    }


    [data-testid="stProgressBar"] > div {
        background: #E6EDF2 !important;
        border-radius: 999px !important;
        height: 7px !important;
    }


    [data-testid="stProgressBar"] > div > div {
        background:
            linear-gradient(
                90deg,
                #2E8C8C,
                #356D8C
            ) !important;
        border-radius: 999px !important;
    }


    /* ====================================================
       Expander
    ==================================================== */

    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E1E8ED !important;
        border-radius: 13px !important;
        overflow: hidden;
        box-shadow:
            0 4px 12px rgba(30, 52, 72, 0.03);
    }


    [data-testid="stExpander"] summary {
        font-weight: 650 !important;
        color: #40566B !important;
    }


    /* ====================================================
       경고 / 안내 박스
    ==================================================== */

    [data-testid="stAlert"] {
        border-radius: 14px !important;
        border: none !important;
    }


    /* ====================================================
       결과 화면 금액 강조
    ==================================================== */

    div[data-testid="stMarkdownContainer"] h1:first-child {
        color: #17324D !important;
    }


    /* ====================================================
       스크롤바
    ==================================================== */

    ::-webkit-scrollbar {
        width: 9px;
    }


    ::-webkit-scrollbar-track {
        background: transparent;
    }


    ::-webkit-scrollbar-thumb {
        background: #CCD7DF;
        border-radius: 999px;
    }


    ::-webkit-scrollbar-thumb:hover {
        background: #AEBCC7;
    }


    /* ====================================================
       모바일
    ==================================================== */

    @media (
        max-width: 640px
    ) {

        .block-container {
            padding-top: 1.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-bottom: 2.5rem;
        }


        h1 {
            font-size: 2.05rem !important;
        }


        h2 {
            font-size: 1.55rem !important;
        }


        h3 {
            font-size: 1.18rem !important;
        }


        [data-testid="stRadio"] {
            padding: 0.85rem;
        }


        .stButton > button {
            min-height: 44px !important;
            font-size: 0.93rem !important;
        }


        .stTextArea textarea {
            min-height: 105px !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 안전 CSS 테마
# GANGWON_SAFE_CSS_V2
# =========================================================

st.markdown(
    """
    <style>

    /* ====================================================
       전체 배경
    ==================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(46, 140, 140, 0.07),
                transparent 28%
            ),
            linear-gradient(
                180deg,
                #F4F8FA 0%,
                #F8FAFB 48%,
                #F5F7F9 100%
            );
    }


    /* ====================================================
       메인 콘텐츠
    ==================================================== */

    .block-container {
        max-width: 760px !important;

        padding-top: 2.7rem !important;
        padding-bottom: 4rem !important;
    }


    /* ====================================================
       최상단 제목
    ==================================================== */

    h1 {
        color: #153652 !important;

        font-size: 2.65rem !important;
        font-weight: 850 !important;

        line-height: 1.16 !important;

        letter-spacing: -0.05em !important;

        margin-bottom: 0.8rem !important;

        text-shadow:
            0 1px 0 rgba(255,255,255,0.75);
    }


    /* 제목 아래 첫 설명문 */
    h1 + div p {
        color: #52677A !important;

        font-size: 1rem !important;

        line-height: 1.75 !important;
    }


    /* ====================================================
       중간 제목
    ==================================================== */

    h3 {
        color: #173A56 !important;

        font-size: 1.3rem !important;
        font-weight: 780 !important;

        letter-spacing: -0.03em !important;

        margin-top: 1.55rem !important;
        margin-bottom: 0.75rem !important;
    }


    /* ====================================================
       일반 본문
    ==================================================== */

    p {
        color: #33495D;
        line-height: 1.7;
    }


    /* ====================================================
       라디오 선택 영역
    ==================================================== */

    [data-testid="stRadio"] {

        background:
            rgba(255,255,255,0.96) !important;

        border:
            1px solid #DCE6EC !important;

        border-radius:
            18px !important;

        padding:
            1.05rem 1.15rem !important;

        margin-top:
            0.35rem !important;

        margin-bottom:
            1.95rem !important;

        box-shadow:
            0 10px 28px rgba(30, 56, 76, 0.075)
            !important;
    }


    [data-testid="stRadio"] label {

        color:
            #344B60 !important;

        font-weight:
            650 !important;

        padding:
            0.2rem 0.1rem;
    }


    /* 선택된 radio의 포인트 */
    [data-testid="stRadio"] input:checked + div {

        color:
            #1B696A !important;

        font-weight:
            750 !important;
    }


    /* ====================================================
       입력 영역
    ==================================================== */

    .stTextArea {

        margin-top:
            0.2rem;
    }


    .stTextArea textarea {

        background:
            rgba(255,255,255,0.98) !important;

        border:
            1px solid #D6E1E8 !important;

        border-radius:
            18px !important;

        min-height:
            120px !important;

        padding:
            1.1rem 1.15rem !important;

        color:
            #24394C !important;

        font-size:
            1rem !important;

        line-height:
            1.65 !important;

        box-shadow:
            0 10px 26px rgba(30, 56, 76, 0.065)
            !important;

        transition:
            border-color 0.2s ease,
            box-shadow 0.2s ease,
            transform 0.2s ease !important;
    }


    .stTextArea textarea:focus {

        border-color:
            #2F8C8B !important;

        box-shadow:
            0 0 0 4px rgba(47, 140, 139, 0.11),
            0 12px 28px rgba(30, 56, 76, 0.08)
            !important;
    }


    .stTextArea textarea::placeholder {

        color:
            #9AA9B5 !important;
    }


    /* ====================================================
       caption
    ==================================================== */

    [data-testid="stCaptionContainer"] {

        color:
            #7D8D99 !important;

        font-size:
            0.88rem !important;

        line-height:
            1.55 !important;
    }


    /* ====================================================
       기본 버튼
    ==================================================== */

    .stButton > button {

        min-height:
            47px !important;

        border-radius:
            13px !important;

        border:
            1px solid #D6E1E8 !important;

        background:
            rgba(255,255,255,0.96) !important;

        color:
            #29455B !important;

        font-weight:
            700 !important;

        letter-spacing:
            -0.015em !important;

        transition:
            all 0.18s ease !important;
    }


    .stButton > button:hover {

        border-color:
            #2E8C8C !important;

        background:
            #F0F8F7 !important;

        color:
            #176867 !important;

        transform:
            translateY(-1px);

        box-shadow:
            0 8px 18px rgba(46, 140, 140, 0.11);
    }


    /* ====================================================
       메인 버튼
    ==================================================== */

    button[data-testid="baseButton-primary"] {

        min-height:
            54px !important;

        border:
            none !important;

        border-radius:
            15px !important;

        background:
            linear-gradient(
                135deg,
                #153954 0%,
                #1D5770 58%,
                #267273 100%
            ) !important;

        color:
            #FFFFFF !important;

        font-size:
            1rem !important;

        font-weight:
            780 !important;

        box-shadow:
            0 12px 26px rgba(21, 57, 84, 0.21)
            !important;
    }


    button[data-testid="baseButton-primary"]:hover {

        background:
            linear-gradient(
                135deg,
                #1A4662 0%,
                #23687B 58%,
                #2A8080 100%
            ) !important;

        color:
            #FFFFFF !important;

        transform:
            translateY(-1px);

        box-shadow:
            0 15px 30px rgba(21, 57, 84, 0.25)
            !important;
    }


    /* ====================================================
       진행바
    ==================================================== */

    [data-testid="stProgressBar"] > div {

        height:
            7px !important;

        background:
            #DFE7EC !important;

        border-radius:
            999px !important;
    }


    [data-testid="stProgressBar"] > div > div {

        background:
            linear-gradient(
                90deg,
                #2E8C8C,
                #236783
            ) !important;

        border-radius:
            999px !important;
    }


    /* ====================================================
       결과 화면 expander
    ==================================================== */

    [data-testid="stExpander"] {

        border:
            1px solid #DDE6EC !important;

        border-radius:
            14px !important;

        background:
            rgba(255,255,255,0.96) !important;

        box-shadow:
            0 6px 16px rgba(30, 56, 76, 0.04)
            !important;

        overflow:
            hidden !important;
    }


    [data-testid="stExpander"] summary {

        color:
            #40586C !important;

        font-weight:
            680 !important;
    }


    /* ====================================================
       구분선
    ==================================================== */

    hr {

        border:
            none !important;

        border-top:
            1px solid #DDE6EC !important;

        margin-top:
            1.7rem !important;

        margin-bottom:
            1.7rem !important;
    }


    /* ====================================================
       결과 화면 제목 강조
    ==================================================== */

    div[data-testid="stMarkdownContainer"] h1 {

        color:
            #153652 !important;
    }


    div[data-testid="stMarkdownContainer"] h2 {

        color:
            #193E59 !important;

        font-weight:
            790 !important;
    }


    /* ====================================================
       입력/선택 사이 공간 확보
    ==================================================== */

    div[data-testid="stTextArea"] {

        margin-bottom:
            0.35rem;
    }


    /* ====================================================
       모바일
    ==================================================== */

    @media (max-width: 640px) {

        .block-container {

            padding-top:
                1.5rem !important;

            padding-left:
                1rem !important;

            padding-right:
                1rem !important;

            padding-bottom:
                2.5rem !important;
        }


        h1 {

            font-size:
                2.05rem !important;
        }


        h3 {

            font-size:
                1.17rem !important;
        }


        [data-testid="stRadio"] {

            padding:
                0.8rem !important;

            border-radius:
                15px !important;
        }


        .stTextArea textarea {

            min-height:
                108px !important;

            border-radius:
                15px !important;
        }


        .stButton > button {

            min-height:
                45px !important;

            font-size:
                0.92rem !important;
        }


        button[data-testid="baseButton-primary"] {

            min-height:
                51px !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. Session State
# =========================================================

if "screen" not in st.session_state:
    st.session_state.screen = "search"

if "results" not in st.session_state:
    st.session_state.results = []

if "result_index" not in st.session_state:
    st.session_state.result_index = 0

if "original_question" not in st.session_state:
    st.session_state.original_question = ""

if "selected_support" not in st.session_state:
    st.session_state.selected_support = "전체"

if "clarify_question" not in st.session_state:
    st.session_state.clarify_question = ""

if "clarify_support" not in st.session_state:
    st.session_state.clarify_support = "전체"


# =========================================================
# 4. Gemini API
# =========================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:

    st.error(
        "GEMINI_API_KEY가 설정되어 있지 않습니다."
    )

    st.stop()


gemini_client = genai.Client(
    api_key=API_KEY
)


# =========================================================
# 5. 임베딩 모델
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


embedding_model = load_embedding_model()


# =========================================================
# 6. ChromaDB
# =========================================================

@st.cache_resource
def load_collection():

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


try:

    collection = load_collection()

except Exception as e:

    st.error(
        "정책자료를 불러오지 못했습니다."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# 7. 전체 DB
# =========================================================

@st.cache_resource
def load_all_records():

    result = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    return (
        result.get("documents", []),
        result.get("metadatas", []),
    )


all_documents, all_metadatas = (
    load_all_records()
)


# =========================================================
# 8. 페이지 인덱스
#
# 검색된 청크의 앞·현재·뒤 페이지를
# Gemini가 함께 볼 수 있도록 준비
# =========================================================

@st.cache_resource
def build_page_index():

    page_index = defaultdict(list)

    for document, metadata in zip(
        all_documents,
        all_metadatas
    ):

        if not document:
            continue

        metadata = metadata or {}

        source = metadata.get(
            "source",
            "출처 미상"
        )

        page = metadata.get(
            "page"
        )

        if page is None:
            continue

        try:

            page = int(page)

        except Exception:

            continue

        page_index[
            (source, page)
        ].append(document)

    return page_index


page_index = build_page_index()


# =========================================================
# 9. 텍스트 정규화
# =========================================================

def normalize_text(text):

    if text is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text).lower()
    ).strip()


# =========================================================
# 10. 질문 키워드
# =========================================================

def extract_keywords(question):

    words = re.findall(
        r"[가-힣A-Za-z0-9&]+",
        question.lower()
    )

    stopwords = {
        "무엇인가요",
        "무엇",
        "어떤",
        "있나요",
        "있습니까",
        "알려주세요",
        "알려줘",
        "받을",
        "받는",
        "있는",
        "위한",
        "하려는",
        "하고",
        "대한",
        "관련",
        "지원",
        "지원사업",
        "사업",
        "기업",
        "중소기업",
        "정책",
        "가능한",
        "가능",
        "수",
        "것",
        "좀",
    }

    keywords = []

    for word in words:

        if len(word) < 2:
            continue

        if word in stopwords:
            continue

        if word not in keywords:
            keywords.append(word)

    return keywords


# =========================================================
# 11. 정책 분야
# =========================================================

INTENT_GROUPS = {

    "창업": [
        "창업",
        "예비창업",
        "초기창업",
        "스타트업",
        "청년창업",
        "창업기업",
    ],

    "소상공인": [
        "소상공인",
        "소공인",
        "자영업",
        "자영업자",
    ],

    "고용": [
        "고용",
        "채용",
        "직원",
        "근로자",
        "인력",
        "인건비",
        "장려금",
    ],

    "AI": [
        "ai",
        "인공지능",
        "ax",
        "ai agent",
        "ai에이전트",
        "머신러닝",
        "딥러닝",
        "온디바이스",
    ],

    "스마트제조": [
        "스마트공장",
        "스마트제조",
        "자동화",
        "공정혁신",
        "제조혁신",
        "자율제조",
    ],

    "기술개발": [
        "기술개발",
        "연구개발",
        "r&d",
        "신제품",
        "기술혁신",
    ],

    "수출": [
        "수출",
        "해외진출",
        "글로벌",
        "해외시장",
        "수출기업",
    ],

    "재창업": [
        "재창업",
        "재도전",
        "폐업",
        "재기",
    ],

    "시설": [
        "시설",
        "시설투자",
        "설비",
        "생산설비",
    ],
}


def detect_intents(question):

    q = normalize_text(question)

    detected = []

    for intent, terms in (
        INTENT_GROUPS.items()
    ):

        if any(
            term in q
            for term in terms
        ):

            detected.append(intent)

    return list(
        dict.fromkeys(detected)
    )


# =========================================================
# 12. 지원 목적
# =========================================================

PURPOSE_GROUPS = {

    "신규채용": [
        "새로 채용",
        "신규 채용",
        "신규채용",
        "직원을 뽑",
        "직원 뽑",
        "사람을 뽑",
        "채용하려",
        "채용하고",
        "새 직원",
    ],

    "도입·활용": [
        "도입",
        "활용",
        "적용",
        "구축",
        "자동화",
        "고도화",
        "스마트공장",
    ],

    "개발·R&D": [
        "개발",
        "연구개발",
        "r&d",
        "기술개발",
        "제품개발",
        "신제품",
    ],

    "사업화": [
        "사업화",
        "상용화",
        "실증",
        "시제품",
        "인증",
        "시장진출",
    ],

    "자금조달": [
        "대출",
        "융자",
        "자금",
        "금리",
        "보증",
        "운전자금",
        "시설자금",
    ],
}


def detect_purposes(question):

    q = normalize_text(question)

    detected = []

    for purpose, terms in (
        PURPOSE_GROUPS.items()
    ):

        if any(
            term in q
            for term in terms
        ):

            detected.append(purpose)

    return list(
        dict.fromkeys(detected)
    )


# =========================================================
# 13. 지역
# =========================================================

REGION_TERMS = {

    "강원": [
        "강원",
        "강원특별자치도",
        "춘천",
        "원주",
        "강릉",
        "속초",
        "동해",
        "삼척",
        "태백",
        "홍천",
        "횡성",
        "영월",
        "평창",
        "정선",
        "철원",
        "화천",
        "양구",
        "인제",
        "고성",
        "양양",
    ],

    "서울": ["서울"],
    "경기": ["경기", "경기도"],
    "인천": ["인천"],
    "충북": ["충북", "충청북도"],
    "충남": ["충남", "충청남도"],
    "대전": ["대전"],
    "세종": ["세종"],
    "전북": ["전북", "전북특별자치도"],
    "전남": ["전남", "전라남도"],
    "광주": ["광주"],
    "경북": ["경북", "경상북도"],
    "경남": ["경남", "경상남도"],
    "대구": ["대구"],
    "부산": ["부산"],
    "울산": ["울산"],
    "제주": ["제주"],
}


def detect_region(question):

    q = normalize_text(question)

    for region, terms in (
        REGION_TERMS.items()
    ):

        if any(
            term in q
            for term in terms
        ):

            return region

    return None


# =========================================================
# 14. 특수조건
#
# 질문에 없는 특수조건이 있으면 감점
# =========================================================

SPECIAL_CONDITIONS = {

    "육아·출산·대체인력": {
        "terms": [
            "육아휴직",
            "육아기",
            "출산",
            "출산휴가",
            "대체인력",
            "출산전후휴가",
        ],
        "penalty": 0.25,
    },

    "외국인력": {
        "terms": [
            "외국인력",
            "외국인 근로자",
            "외국인근로자",
            "외국인 고용",
            "고용허가제",
            "e-9",
        ],
        "penalty": 0.30,
    },

    "장애인": {
        "terms": [
            "장애인",
            "장애인 고용",
            "장애인근로자",
        ],
        "penalty": 0.30,
    },

    "신설·증설": {
        "terms": [
            "신설",
            "증설",
            "신·증설",
            "신증설",
            "공장증설",
            "공장 신설",
        ],
        "penalty": 0.45,
    },

    "청년": {
        "terms": [
            "청년",
            "청년고용",
            "청년 고용",
        ],
        "penalty": 0.65,
    },

    "고령자": {
        "terms": [
            "고령자",
            "60세 이상",
            "60세이상",
            "계속고용",
        ],
        "penalty": 0.30,
    },

    "재창업·폐업": {
        "terms": [
            "재창업",
            "재도전",
            "폐업",
            "채무조정",
        ],
        "penalty": 0.35,
    },
}


def special_condition_multiplier(
    document,
    source,
    question,
):

    combined_text = normalize_text(
        f"{document} {source}"
    )

    question_text = normalize_text(
        question
    )

    multiplier = 1.0


    for _, config in (
        SPECIAL_CONDITIONS.items()
    ):

        terms = config["terms"]

        document_has = any(
            term in combined_text
            for term in terms
        )

        question_has = any(
            term in question_text
            for term in terms
        )

        if (
            document_has
            and not question_has
        ):

            multiplier *= config[
                "penalty"
            ]

    return multiplier


# =========================================================
# 15. 신규채용 의도 필터
# =========================================================

NEW_HIRE_POSITIVE_TERMS = [
    "신규 채용",
    "신규채용",
    "신규 고용",
    "신규고용",
    "채용지원",
    "채용 지원",
    "실업자를 고용",
    "근로자를 고용",
    "고용촉진",
    "고용장려금",
    "채용 1인당",
    "신규 채용지원",
]


NEW_HIRE_NEGATIVE_GROUPS = {

    "고용유지": {
        "terms": [
            "고용유지지원금",
            "고용유지조치",
            "휴업",
            "휴직수당",
            "고용 유지",
        ],
        "penalty": 0.30,
    },

    "정규직전환": {
        "terms": [
            "정규직 전환",
            "정규직전환",
            "기간제 근로자",
        ],
        "penalty": 0.45,
    },

    "계속고용": {
        "terms": [
            "계속고용",
            "정년 연장",
            "정년연장",
            "재고용",
        ],
        "penalty": 0.25,
    },

    "산재복귀": {
        "terms": [
            "산재장해인",
            "직장복귀지원금",
            "원직장 복귀",
        ],
        "penalty": 0.20,
    },

    "근로복지": {
        "terms": [
            "근로복지기금",
            "복지기금",
            "복지비용",
        ],
        "penalty": 0.40,
    },
}


def is_new_hire_question(
    question
):

    q = normalize_text(question)

    return any(
        term in q
        for term in PURPOSE_GROUPS[
            "신규채용"
        ]
    )


def new_hire_multiplier(
    document,
    source,
    question,
):

    if not is_new_hire_question(
        question
    ):

        return 1.0


    text = normalize_text(
        f"{document} {source}"
    )

    multiplier = 1.0


    positive_matches = sum(
        1
        for term in NEW_HIRE_POSITIVE_TERMS
        if term in text
    )


    if positive_matches >= 1:

        multiplier *= 1.60


    if positive_matches >= 2:

        multiplier *= 1.25


    for _, config in (
        NEW_HIRE_NEGATIVE_GROUPS.items()
    ):

        if any(
            term in text
            for term in config["terms"]
        ):

            multiplier *= config[
                "penalty"
            ]


    return multiplier


# =========================================================
# 16. 금전지원 우선
# =========================================================

MONEY_TERMS = [
    "장려금",
    "지원금",
    "보조금",
    "인건비",
    "임금 지원",
    "임금지원",
    "지원한도",
    "지원 한도",
    "지원수준",
    "만원",
    "억원",
    "지급",
]


SERVICE_TERMS = [
    "컨설팅",
    "상담",
    "알선",
    "채용대행",
    "동행면접",
    "구인·구직",
    "취업지원 서비스",
]


def money_support_multiplier(
    document,
    question,
):

    text = normalize_text(document)

    money_matches = sum(
        1
        for term in MONEY_TERMS
        if term in text
    )

    service_matches = sum(
        1
        for term in SERVICE_TERMS
        if term in text
    )

    multiplier = 1.0


    if money_matches >= 1:

        multiplier *= 1.25


    if money_matches >= 3:

        multiplier *= 1.15


    if (
        service_matches >= 1
        and money_matches == 0
    ):

        multiplier *= 0.45


    return multiplier


# =========================================================
# 17. 최신·변경 공고 우대
# =========================================================

def latest_source_multiplier(
    source
):

    text = normalize_text(source)

    multiplier = 1.0


    if "변경공고" in text:

        multiplier *= 1.20


    if "수정" in text:

        multiplier *= 1.10


    return multiplier


# =========================================================
# 18. 분야 점수
# =========================================================

def intent_score(
    document,
    intents
):

    if not intents:
        return 0.0

    text = normalize_text(document)

    score = 0.0


    for intent in intents:

        terms = INTENT_GROUPS.get(
            intent,
            []
        )

        matches = sum(
            1
            for term in terms
            if term in text
        )

        if matches > 0:

            score += 3.0

            score += min(
                matches,
                3
            ) * 0.6


    return score


# =========================================================
# 19. 목적 점수
# =========================================================

def purpose_score(
    document,
    purposes
):

    if not purposes:
        return 0.0

    text = normalize_text(document)

    score = 0.0


    for purpose in purposes:

        terms = PURPOSE_GROUPS.get(
            purpose,
            []
        )

        matches = sum(
            1
            for term in terms
            if term in text
        )

        if matches > 0:

            score += 4.0

            score += min(
                matches,
                3
            ) * 0.7


    return score


# =========================================================
# 20. 지역 점수
# =========================================================

def region_score(
    document,
    source,
    requested_region
):

    if not requested_region:

        return 0.0


    text = normalize_text(document)

    source_text = normalize_text(source)


    requested_terms = REGION_TERMS.get(
        requested_region,
        []
    )


    if any(
        term in text
        for term in requested_terms
    ):

        return 6.0


    local_context_terms = [
        "도내 기업",
        "도내기업",
        "도내 중소기업",
        "도내 소재",
        "지역 내 중소기업",
        "지역내 중소기업",
    ]


    if (
        requested_region == "강원"
        and "강원" in source_text
        and any(
            term in text
            for term in local_context_terms
        )
    ):

        return 5.0


    for region, terms in (
        REGION_TERMS.items()
    ):

        if region == requested_region:
            continue

        if any(
            term in text
            for term in terms
        ):

            return -7.0


    return 0.0


# =========================================================
# 21. 키워드 점수
# =========================================================

def keyword_score(
    document,
    source,
    keywords
):

    if not keywords:

        return 0.0


    document_text = normalize_text(
        document
    )

    source_text = normalize_text(
        source
    )


    score = 0.0

    matched = 0


    for keyword in keywords:

        key = normalize_text(
            keyword
        )

        if key in document_text:

            score += 1.4

            matched += 1


        if key in source_text:

            score += 0.3


    if matched >= 2:

        score += 1.5


    if matched >= 3:

        score += 1.5


    return score


# =========================================================
# 22. 의미검색
# =========================================================

def vector_search(
    question
):

    query_embedding = (
        embedding_model.encode(
            question,
            normalize_embeddings=True,
        )
        .tolist()
    )


    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=min(
            VECTOR_SEARCH_COUNT,
            collection.count()
        ),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )


    output = []


    documents = results.get(
        "documents",
        [[]]
    )[0]


    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]


    distances = results.get(
        "distances",
        [[]]
    )[0]


    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        output.append(
            {
                "document": document,
                "metadata": metadata or {},
                "distance": float(distance),
            }
        )


    return output


# =========================================================
# 23. 후보 검색
# =========================================================

def build_ranked_candidates(
    question,
    support_type
):

    keywords = extract_keywords(
        question
    )

    intents = detect_intents(
        question
    )

    purposes = detect_purposes(
        question
    )

    requested_region = detect_region(
        question
    )


    search_question = question


    if support_type == "지원금":

        search_question += (
            " 지원금 보조금 장려금 "
            "인건비 사업비 비용지원"
        )


    elif support_type == "융자":

        search_question += (
            " 융자 대출 정책자금 금리"
        )


    elif support_type == "보증":

        search_question += (
            " 보증 신용보증 협약보증"
        )


    elif support_type == "R&D":

        search_question += (
            " 연구개발 R&D 기술개발 연구비"
        )


    if is_new_hire_question(
        question
    ):

        search_question += (
            " 신규채용 신규 고용 "
            "사업주 고용장려금 "
            "채용 인건비 지원"
        )


    vector_results = vector_search(
        search_question
    )


    vector_distance_map = {}


    for item in vector_results:

        normalized = normalize_text(
            item["document"]
        )

        vector_distance_map[
            normalized
        ] = item[
            "distance"
        ]


    candidates = []

    seen = set()


    for document, metadata in zip(
        all_documents,
        all_metadatas
    ):

        if not document:

            continue


        normalized_document = (
            normalize_text(document)
        )


        if normalized_document in seen:

            continue


        seen.add(
            normalized_document
        )


        metadata = metadata or {}


        source = metadata.get(
            "source",
            "출처 미상"
        )


        distance = (
            vector_distance_map.get(
                normalized_document
            )
        )


        semantic_score = 0.0


        if distance is not None:

            semantic_score = max(
                0.0,
                1.0 - distance
            ) * 9.0


        k_score = keyword_score(
            document,
            source,
            keywords
        )


        i_score = intent_score(
            document,
            intents
        )


        p_score = purpose_score(
            document,
            purposes
        )


        r_score = region_score(
            document,
            source,
            requested_region
        )


        total_score = (
            semantic_score
            + k_score
            + i_score
            + p_score
            + r_score
        )


        if (
            intents
            and i_score <= 0
        ):

            total_score -= 4.0


        if (
            purposes
            and p_score <= 0
        ):

            total_score -= 3.0


        if r_score < 0:

            total_score -= 3.0


        # ================================================
        # 검증한 필터 적용
        # ================================================

        total_score *= (
            special_condition_multiplier(
                document,
                source,
                question,
            )
        )


        total_score *= (
            new_hire_multiplier(
                document,
                source,
                question,
            )
        )


        total_score *= (
            money_support_multiplier(
                document,
                question,
            )
        )


        total_score *= (
            latest_source_multiplier(
                source
            )
        )


        # ================================================
        # 선택 지원유형 우대
        # ================================================

        doc_text = normalize_text(
            document
        )


        if support_type == "지원금":

            if any(
                term in doc_text
                for term in [
                    "지원금",
                    "보조금",
                    "장려금",
                    "인건비",
                    "정부지원",
                    "지원비율",
                    "사업비",
                    "지원한도",
                ]
            ):

                total_score += 4.0


        elif support_type == "융자":

            if any(
                term in doc_text
                for term in [
                    "융자",
                    "대출",
                    "금리",
                    "정책자금",
                ]
            ):

                total_score += 4.0


        elif support_type == "보증":

            if any(
                term in doc_text
                for term in [
                    "보증",
                    "보증료",
                    "보증한도",
                    "신용보증",
                ]
            ):

                total_score += 4.0


        elif support_type == "R&D":

            if any(
                term in doc_text
                for term in [
                    "r&d",
                    "연구개발",
                    "기술개발",
                    "정부출연금",
                    "연구비",
                ]
            ):

                total_score += 4.0


        if total_score <= 0:

            continue


        candidates.append(
            {
                "document": document,
                "metadata": metadata,
                "total_score": total_score,
            }
        )


    candidates.sort(
        key=lambda x: x[
            "total_score"
        ],
        reverse=True
    )


    selected = []

    page_seen = set()

    source_counts = defaultdict(int)


    for item in candidates:

        metadata = item[
            "metadata"
        ]


        source = metadata.get(
            "source",
            "출처 미상"
        )


        page = metadata.get(
            "page",
            "페이지 미상"
        )


        page_key = (
            source,
            page
        )


        if page_key in page_seen:

            continue


        if source_counts[
            source
        ] >= 5:

            continue


        page_seen.add(
            page_key
        )


        source_counts[
            source
        ] += 1


        selected.append(
            item
        )


        if len(
            selected
        ) >= FINAL_CONTEXT_COUNT:

            break


    return selected


# =========================================================
# 24. 앞·현재·뒤 페이지
# =========================================================

def get_expanded_context(
    source,
    page
):

    try:

        page = int(page)

    except Exception:

        return ""


    combined = []


    for target_page in [
        page - 1,
        page,
        page + 1,
    ]:

        if target_page < 1:

            continue


        chunks = page_index.get(
            (source, target_page),
            []
        )


        if not chunks:

            continue


        combined.append(
            f"""
[페이지 {target_page}]

{" ".join(chunks)}
"""
        )


    return "\n".join(
        combined
    )


# =========================================================
# 25. JSON 정리
# =========================================================

def clean_json_text(text):

    text = text.strip()


    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )


    text = re.sub(
        r"^```\s*",
        "",
        text
    )


    text = re.sub(
        r"\s*```$",
        "",
        text
    )


    return text.strip()


# =========================================================
# 26. 금액 문자열 → 정렬용 숫자
# =========================================================

def amount_to_number(
    amount_text
):

    if not amount_text:

        return -1


    text = str(
        amount_text
    ).replace(
        ",",
        ""
    )


    if "확인 필요" in text:

        return -1


    # 월 지급액도 일단 표시금액 기준으로 사용
    eok_match = re.search(
        r"(\d+(?:\.\d+)?)\s*억",
        text
    )


    man_match = re.search(
        r"(\d+(?:\.\d+)?)\s*만",
        text
    )


    if eok_match:

        return int(
            float(
                eok_match.group(1)
            )
            * 100_000_000
        )


    if man_match:

        return int(
            float(
                man_match.group(1)
            )
            * 10_000
        )


    return -1


# =========================================================
# 27. 최종 결과 정렬
# =========================================================

TYPE_PRIORITY = {

    "지원금·보조금": 1,

    "인건비·고용지원": 1,

    "R&D": 2,

    "보증": 3,

    "융자·대출": 4,

    "컨설팅·서비스": 5,

    "기타": 6,
}


def sort_results(
    results,
    support_type
):

    def sort_key(item):

        amount = amount_to_number(
            item.get(
                "amount",
                ""
            )
        )


        item_type = item.get(
            "type",
            "기타"
        )


        if support_type == "전체":

            return (
                TYPE_PRIORITY.get(
                    item_type,
                    9
                ),

                0 if amount >= 0 else 1,

                -amount,
            )


        return (
            0 if amount >= 0 else 1,

            -amount,
        )


    return sorted(
        results,
        key=sort_key
    )


# =========================================================
# 28. Gemini 결과 생성
# =========================================================

def generate_results(
    question,
    support_type
):

    ranked_results = (
        build_ranked_candidates(
            question,
            support_type
        )
    )


    if not ranked_results:

        return []


    context_parts = []


    for index, item in enumerate(
        ranked_results,
        start=1
    ):

        metadata = item[
            "metadata"
        ]


        source = metadata.get(
            "source",
            "출처 미상"
        )


        page = metadata.get(
            "page",
            "페이지 미상"
        )


        expanded_text = (
            get_expanded_context(
                source,
                page
            )
        )


        context_parts.append(
            f"""
========================================
후보자료 {index}
========================================

파일:
{source}

검색 중심 페이지:
{page}

앞·현재·뒤 페이지:

{expanded_text}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    # =====================================================
    # 지원 유형 규칙
    # =====================================================

    if support_type == "지원금":

        support_rule = """
지원금·보조금·장려금·인건비 등
실제 비용지원 성격의 사업만 선정하세요.

융자와 보증은 제외하세요.

상담·교육·컨설팅만 제공하는 사업은 제외하세요.
"""


    elif support_type == "융자":

        support_rule = """
융자·대출·정책자금 사업만 선정하세요.

지원금, 보증, 단순 R&D는 제외하세요.
"""


    elif support_type == "보증":

        support_rule = """
신용보증, 협약보증 등
보증 성격의 사업만 선정하세요.
"""


    elif support_type == "R&D":

        support_rule = """
연구개발비 또는 기술개발비를
지원하는 R&D 사업만 선정하세요.
"""


    else:

        support_rule = """
사용자의 상황에 가장 직접적으로 도움이 되는 사업을
종류와 관계없이 최대 3개 선정하세요.

실제 비용 지원이 가능한 사업을 우선 검토하세요.

우선순위는 다음과 같습니다.

1. 지원금·보조금
2. 인건비·고용지원
3. R&D
4. 보증
5. 융자·대출
6. 컨설팅·서비스
"""


    new_hire_rule = ""


    if is_new_hire_question(
        question
    ):

        new_hire_rule = """
사용자가 '새로운 직원을 채용'하려는 상황입니다.

가장 중요합니다.

신규채용을 직접 지원하는
장려금·인건비 지원사업을 우선하세요.

다음과 같은 사업은 사용자가 해당 조건을
직접 언급하지 않았다면 추천하지 마세요.

- 육아휴직 대체인력
- 기존 직원 고용유지
- 고령자 계속고용
- 정규직 전환
- 장애인 전용
- 외국인력 전용
- 산재근로자 직장복귀
- 공장 신설·증설을 전제로 하는 지원

단순히 '고용'이라는 단어가 있다고
신규채용 지원으로 판단하면 안 됩니다.
"""


    prompt = f"""
당신은 정책정보에 익숙하지 않은
소상공인과 소규모 사업자를 위한
정책지원 안내 도우미입니다.

반드시 아래 PDF 근거자료만 사용하세요.


사용자의 선택:
{support_type}


사용자의 질문:
{question}


==================================================
지원 유형
==================================================

{support_rule}


==================================================
질문 목적 특별 규칙
==================================================

{new_hire_rule}


==================================================
반드시 지켜야 할 규칙
==================================================

1. 사용자의 지역, 업종, 목적과 직접 관련된 사업만 선정하세요.

2. 최대 3개까지만 선정하세요.

3. 적합한 사업이 1개 또는 2개면
억지로 3개를 채우지 마세요.

4. 다른 지역에만 해당되는 사업은 제외하세요.

5. 질문에 없는 특별한 자격조건을
임의로 가정하면 안 됩니다.

6. 동일한 사업이 여러 페이지에서 검색되더라도
최종 결과에는 한 번만 표시하세요.

7. 사업명이 같거나 실질적으로 동일한 사업은
중복 추천하지 마세요.

8. 변경공고와 최초공고의 내용이 충돌할 경우
변경공고 또는 수정공고의 내용을 우선하세요.

9. 앞·현재·뒤 페이지를 모두 읽고
지원금액과 지원조건을 확인하세요.

10. 지원금액이 명확한 사업을
금액을 확인할 수 없는 사업보다 우선하세요.

11. 사업 전체 예산을
개별 기업이 받을 수 있는 금액처럼 표현하면 안 됩니다.

12. 융자한도는 지원금이 아닙니다.

13. 단순 상담이나 컨설팅은
실제 금전지원으로 표현하지 마세요.


==================================================
사실정보 정확성 규칙
==================================================

아래 규칙은 매우 중요합니다.

14. 지원금액, 대출한도, 보증한도, 지원비율,
지원기간, 지원대상, 문의처는
PDF 근거자료에 있는 정보만 사용하세요.

15. 문의처의 전화번호는 반드시 PDF 근거자료에
실제로 적혀 있는 경우에만 그대로 작성하세요.

16. 전화번호의 일부 숫자를 추측하거나,
다른 자료의 전화번호와 조합하거나,
기관의 대표번호를 임의로 만들어서는 안 됩니다.

17. 기관명은 확인되지만 전화번호를
근거자료에서 확인할 수 없다면:

"기관명 · 세부 공고 확인"

형태로 작성하세요.

18. 기관명도 명확하지 않다면:

"세부 공고 확인"

이라고 작성하세요.

19. PDF에 적힌 전화번호는 숫자와 하이픈을
원문 그대로 유지하세요.

20. 지원금액도 절대 추측하지 마세요.

21. 사업 전체 예산,
예를 들어 "사업예산 100억원",
"총사업비 50억원" 등의 숫자를
개별 기업이 받을 수 있는 지원금액으로
표현해서는 안 됩니다.

22. 기업당, 과제당, 인당, 사업장당 등의
개별 지원한도가 명확한 경우에만
amount에 숫자를 작성하세요.

23. 지원비율만 있고 최대금액이 없는 경우
없는 최대금액을 만들어내지 마세요.

예:
"비용의 70% 이내"

처럼 근거자료에 확인되는 범위만 표현하세요.

24. 금액이 근거자료에서 명확하게 확인되지 않으면:

"지원금액 확인 필요"

라고 작성하세요.

25. 지원대상 역시 PDF에 없는 조건을
상식이나 추론으로 추가하면 안 됩니다.

26. 질문자의 지역이 해당 사업의 지원대상이라는
근거가 불충분하면 해당 사업을 추천하지 마세요.

27. 서로 다른 PDF의 숫자나 조건을 합쳐서
하나의 사업 조건처럼 작성해서는 안 됩니다.

28. 추천이유에는 설명을 쉽게 바꾸는 것은 가능하지만,
새로운 자격조건이나 혜택을 추가해서는 안 됩니다.


==================================================
사업 유형
==================================================

type 값은 아래 중 하나만 사용하세요.

지원금·보조금
인건비·고용지원
융자·대출
보증
R&D
컨설팅·서비스
기타


==================================================
JSON 출력
==================================================

반드시 JSON만 반환하세요.

마크다운을 사용하지 마세요.

형식:

{{
  "results": [
    {{
      "amount": "월 60~80만 원",
      "name": "사업 공식 명칭",
      "type": "인건비·고용지원",
      "reason": "이 사업을 추천하는 이유를 쉬운 한 문장으로 작성",
      "target": "핵심 신청 대상 한 줄",
      "contact": "PDF에서 실제 확인된 기관명과 전화번호. 전화번호가 없으면 기관명 · 세부 공고 확인",
      "source": "PDF 파일명",
      "page": "120"
    }}
  ]
}}


==================================================
금액 표기
==================================================

지원금:
"최대 4,000만 원"

월별 장려금:
"월 60~80만 원"

기간까지 명확하면:
"월 60만 원 × 최대 12개월"

융자:
"대출한도 최대 100억 원"

보증:
"보증한도 최대 8억 원"

R&D:
"연구개발비 최대 3억 원"

정말 찾을 수 없는 경우에만:
"지원금액 확인 필요"


==================================================
문장 길이
==================================================

reason:
반드시 한 문장.

target:
한 줄.

contact:
한 줄.


==================================================
PDF 근거자료
==================================================

{context}
"""


    response = (
        gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
    )


    if not response.text:

        return []


    cleaned = clean_json_text(
        response.text
    )


    data = json.loads(
        cleaned
    )


    results = data.get(
        "results",
        []
    )


    return sort_results(
        results,
        support_type
    )


# =========================================================
# 애매한 질문 판별
#
# 사용자가 "장사가 어렵다", "도움이 필요하다" 정도만
# 말하고 실제 필요한 지원 목적을 밝히지 않았을 때
# 바로 정책을 추측하지 않고 한 번 더 물어봅니다.
# =========================================================

def needs_clarification(
    question
):

    q = normalize_text(
        question
    )


    # -----------------------------------------------------
    # 사용자가 이미 목적을 명확히 말한 경우
    # 추가 질문이 필요하지 않음
    # -----------------------------------------------------

    clear_purpose_terms = [

        # 자금
        "운영자금",
        "운전자금",
        "대출",
        "융자",
        "보증",
        "금리",
        "자금이 필요",
        "자금 마련",
        "자금마련",
        "담보",
        "담보가 부족",
        "신용보증",
        "보증이 필요",

        # 채용
        "채용",
        "직원",
        "인건비",
        "고용",

        # 홍보 / 판로
        "마케팅",
        "홍보",
        "온라인 판매",
        "온라인판매",
        "판로",
        "광고",
        "브랜딩",
        "쇼핑몰",

        # 시설
        "시설개선",
        "시설 개선",
        "리모델링",
        "간판",
        "설비",
        "장비",
        "기계",
        "키오스크",

        # 기술
        "ai",
        "인공지능",
        "스마트공장",
        "기술개발",
        "연구개발",
        "r&d",

        # 수출
        "수출",
        "해외진출",
        "해외 진출",
        "해외시장",

        # 창업
        "창업",
        "예비창업",
        "가게를 열",
        "사업을 시작",

    ]


    if any(
        term in q
        for term in clear_purpose_terms
    ):

        return False


    # -----------------------------------------------------
    # 목적은 없고 어려움만 표현한 질문
    # -----------------------------------------------------

    vague_problem_terms = [
        "장사가 어렵",
        "장사가 안",
        "매출이 안",
        "매출이 줄",
        "힘들",
        "어렵",
        "도움",
        "지원이 있을",
        "받을 수 있는 지원",
        "뭐 받을",
        "무슨 지원",
        "혜택이 있",
        "지원받을",
    ]


    return any(
        term in q
        for term in vague_problem_terms
    )


# =========================================================
# 선택한 도움을 원래 질문에 자연스럽게 추가
# =========================================================

def make_clarified_question(
    original_question,
    choice
):

    additions = {

        "사업비·운영비 지원":
            " 특히 사업 운영에 드는 비용을 줄일 수 있는 "
            "지원금, 바우처, 보조금이 필요합니다.",

        "대출·자금조달":
            " 특히 사업 운영에 필요한 운전자금, "
            "경영안정자금, 정책자금 대출이 필요합니다.",

        "직원 인건비":
            " 특히 직원 채용이나 인건비 부담을 "
            "줄일 수 있는 장려금과 지원이 필요합니다.",

        "홍보·마케팅":
            " 특히 고객을 늘리기 위한 홍보, "
            "마케팅, 온라인 판로 지원이 필요합니다.",

        "시설·장비":
            " 특히 매장 시설개선, 장비 교체, "
            "설비 도입 관련 지원이 필요합니다.",

        "보증":
            " 특히 담보가 부족해 자금조달을 위한 "
            "신용보증이나 협약보증이 필요합니다.",

        "잘 모르겠어요":
            " 어떤 종류의 지원이 가장 적합한지 "
            "잘 모르겠으니 현재 상황에서 "
            "실질적으로 도움이 되는 지원을 찾아주세요.",
    }


    return (
        original_question.strip()
        + additions.get(
            choice,
            ""
        )
    )


# =========================================================
# 추가 질문 선택에 따른 실제 검색 지원유형 결정
#
# 운영자금과 시설개선은 지원금뿐 아니라
# 융자·보증 등도 중요한 경우가 많으므로
# "전체" 범위에서 검색합니다.
# =========================================================

def resolve_clarify_support_type(
    choice,
    original_support_type,
):

    # 사업비·운영비 지원
    # 보조금·바우처 중심
    if choice == "사업비·운영비 지원":

        return "지원금"


    # 대출·자금조달
    # 정책자금·운전자금·융자 중심
    if choice == "대출·자금조달":

        return "융자"


    # 직원 인건비
    # 장려금·인건비 지원 중심
    if choice == "직원 인건비":

        return "지원금"


    # 홍보·마케팅
    # 판로·바우처·사업비 중심
    if choice == "홍보·마케팅":

        return "지원금"


    # 시설·장비
    # 보조금과 시설자금이 함께 있을 수 있음
    if choice == "시설·장비":

        return "전체"


    # 보증
    if choice == "보증":

        return "보증"


    # 사용자가 무엇이 필요한지 모르는 경우
    if choice == "잘 모르겠어요":

        return "전체"


    return original_support_type


# =========================================================
# 29. 처음으로
# =========================================================

def go_home():

    st.session_state.screen = "search"

    st.session_state.results = []

    st.session_state.result_index = 0

    st.session_state.original_question = ""

    st.session_state.clarify_question = ""

    st.session_state.clarify_support = "전체"

    st.rerun()


# =========================================================
# 30. 질문 화면
# =========================================================

if st.session_state.screen == "search":

    st.title(
        "강원 소상공인 혜택 도우미"
    )


    st.write(
        "복잡한 지원사업 공고를 직접 찾지 않아도 됩니다. "
        "사업 상황을 알려주시면 확인해볼 만한 지원을 찾아드립니다."
    )


    st.markdown(
        "### 어떤 도움이 필요하세요?"
    )


    support_choice = st.radio(
        "지원 종류",
        [
            "💰 지원금",
            "🏦 대출·융자",
            "🛡 보증",
            "🔬 기술개발",
            "✨ 전체",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )


    SUPPORT_MAP = {

        "💰 지원금":
            "지원금",

        "🏦 대출·융자":
            "융자",

        "🛡 보증":
            "보증",

        "🔬 기술개발":
            "R&D",

        "✨ 전체":
            "전체",
    }


    selected_support_type = (
        SUPPORT_MAP[
            support_choice
        ]
    )


    st.markdown(
        "### 사업 상황을 편하게 적어주세요."
    )


    question = st.text_area(
        "사업 상황",
        placeholder=(
            "예: 원주에서 작은 식당을 운영하고 있는데 "
            "요즘 매출이 줄어 지원을 받고 싶어요."
        ),
        height=120,
        label_visibility="collapsed",
    )


    st.caption(
        "지역, 업종, 현재 상황을 함께 적어주시면 "
        "더 잘 맞는 지원을 찾을 수 있습니다."
    )


    if st.button(
        "내게 맞는 지원 확인하기",
        use_container_width=True,
        type="primary",
    ):

        if not question.strip():

            st.warning(
                "궁금한 내용을 입력해주세요."
            )


        else:

            # =============================================
            # 질문의 목적이 너무 넓으면
            # Gemini를 호출하기 전에 한 번 더 질문
            # =============================================

            if needs_clarification(
                question
            ):

                st.session_state.clarify_question = (
                    question
                )

                st.session_state.clarify_support = (
                    selected_support_type
                )

                st.session_state.screen = (
                    "clarify"
                )

                st.rerun()


            try:

                with st.spinner(
                    "조건에 맞는 지원사업을 찾고 있습니다..."
                ):

                    results = (
                        generate_results(
                            question,
                            selected_support_type
                        )
                    )


                if not results:

                    st.warning(
                        "현재 등록된 자료에서는 "
                        "조건에 맞는 지원사업을 찾지 못했습니다."
                    )


                else:

                    st.session_state.results = (
                        results
                    )

                    st.session_state.result_index = 0

                    st.session_state.original_question = (
                        question
                    )

                    st.session_state.selected_support = (
                        selected_support_type
                    )

                    st.session_state.screen = (
                        "results"
                    )

                    st.rerun()


            except Exception as e:

                error_text = str(e)


                if (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "quota" in error_text.lower()
                    or "rate limit" in error_text.lower()
                ):

                    st.warning(
                        "현재 AI 답변 사용량이 많아 "
                        "잠시 답변을 만들 수 없습니다."
                    )

                    st.info(
                        "잠시 후 다시 시도해주세요. "
                        "저장된 정책자료와 검색 데이터에는 "
                        "문제가 없습니다."
                    )


                else:

                    st.error(
                        "검색 결과를 만드는 중 "
                        "오류가 발생했습니다."
                    )

                    st.code(
                        error_text
                    )


# =========================================================
# 31. 추가 질문 화면
# =========================================================

elif st.session_state.screen == "clarify":

    st.title(
        "조금만 더 알려주세요"
    )


    st.write(
        "더 잘 맞는 지원을 찾기 위해 "
        "지금 가장 필요한 도움을 선택해주세요."
    )


    st.caption(
        "정확한 정책 이름을 모르셔도 됩니다."
    )


    st.divider()


    # -----------------------------------------------------
    # 사용자가 입력했던 내용
    # -----------------------------------------------------

    with st.expander(
        "내가 입력한 내용"
    ):

        st.write(
            st.session_state.clarify_question
        )


    st.markdown(
        "### 지금 가장 필요한 것은 무엇인가요?"
    )


    st.caption(
        "가장 가까운 항목 하나만 선택해주세요."
    )


    col1, col2 = st.columns(2)


    with col1:

        operating_support = st.button(
            "💰 사업비·운영비 지원",
            use_container_width=True,
        )


    with col2:

        financing = st.button(
            "🏦 대출·자금조달",
            use_container_width=True,
        )


    col3, col4 = st.columns(2)


    with col3:

        labor = st.button(
            "👥 직원 인건비",
            use_container_width=True,
        )


    with col4:

        marketing = st.button(
            "📢 홍보·마케팅",
            use_container_width=True,
        )


    col5, col6 = st.columns(2)


    with col5:

        facility = st.button(
            "🏪 시설·장비",
            use_container_width=True,
        )


    with col6:

        guarantee = st.button(
            "🛡 보증",
            use_container_width=True,
        )


    unsure = st.button(
        "잘 모르겠어요",
        use_container_width=True,
    )


    choice = None


    if operating_support:

        choice = "사업비·운영비 지원"


    elif financing:

        choice = "대출·자금조달"


    elif labor:

        choice = "직원 인건비"


    elif marketing:

        choice = "홍보·마케팅"


    elif facility:

        choice = "시설·장비"


    elif guarantee:

        choice = "보증"


    elif unsure:

        choice = "잘 모르겠어요"


    # -----------------------------------------------------
    # 선택 후 실제 검색
    # -----------------------------------------------------

    if choice:

        clarified_question = (
            make_clarified_question(
                st.session_state.clarify_question,
                choice,
            )
        )


        # =============================================
        # 사용자가 추가로 선택한 실제 목적에 따라
        # 검색할 지원유형을 다시 결정합니다.
        # =============================================

        clarified_support_type = (
            resolve_clarify_support_type(
                choice,
                st.session_state.clarify_support,
            )
        )


        try:

            with st.spinner(
                "조건에 맞는 지원사업을 찾고 있습니다..."
            ):

                results = generate_results(
                    clarified_question,
                    clarified_support_type,
                )


            if not results:

                st.warning(
                    "현재 등록된 자료에서는 "
                    "조건에 맞는 지원사업을 찾지 못했습니다."
                )


            else:

                st.session_state.results = (
                    results
                )

                st.session_state.result_index = 0

                st.session_state.original_question = (
                    clarified_question
                )

                st.session_state.selected_support = (
                    clarified_support_type
                )

                st.session_state.screen = (
                    "results"
                )

                st.rerun()


        except Exception as e:

            error_text = str(e)


            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
                or "rate limit" in error_text.lower()
            ):

                st.warning(
                    "현재 AI 답변 사용량이 많아 "
                    "잠시 답변을 만들 수 없습니다."
                )

                st.info(
                    "잠시 후 다시 시도해주세요. "
                    "저장된 정책자료와 검색 데이터에는 "
                    "문제가 없습니다."
                )


            else:

                st.error(
                    "검색 결과를 만드는 중 "
                    "오류가 발생했습니다."
                )

                st.code(
                    error_text
                )


    st.divider()


    if st.button(
        "← 질문 다시 쓰기",
        use_container_width=True,
    ):

        st.session_state.screen = "search"

        st.rerun()


# =========================================================
# 32. 결과 화면
# =========================================================

else:

    results = (
        st.session_state.results
    )


    index = (
        st.session_state.result_index
    )


    total = len(results)


    if not results:

        go_home()


    current = results[
        index
    ]


    st.caption(
        f"추천 결과 {index + 1} / {total}"
    )


    st.progress(
        (index + 1) / total
    )


    # =====================================================
    # 지원 금액
    # =====================================================

    st.markdown(
        f"# {current.get('amount', '지원금액 확인 필요')}"
    )


    # =====================================================
    # 사업명
    # =====================================================

    st.markdown(
        f"## {current.get('name', '지원사업')}"
    )


    # =====================================================
    # 지원 종류
    # =====================================================

    current_type = current.get(
        "type",
        "기타"
    )


    TYPE_LABELS = {

        "지원금·보조금":
            "💰 지원금·보조금",

        "인건비·고용지원":
            "👥 인건비·고용지원",

        "융자·대출":
            "🏦 융자·대출",

        "보증":
            "🛡 보증",

        "R&D":
            "🔬 연구개발 지원",

        "컨설팅·서비스":
            "📋 서비스 지원",

        "기타":
            "지원사업",
    }


    st.caption(
        TYPE_LABELS.get(
            current_type,
            current_type
        )
    )


    st.divider()


    # =====================================================
    # 추천 이유
    # =====================================================

    st.markdown(
        "### 추천이유"
    )


    st.write(
        current.get(
            "reason",
            "확인 필요"
        )
    )


    # =====================================================
    # 지원 대상
    # =====================================================

    st.markdown(
        "### 지원대상"
    )


    st.write(
        current.get(
            "target",
            "확인 필요"
        )
    )


    # =====================================================
    # 문의처
    # =====================================================

    st.markdown(
        "### 문의처"
    )


    st.write(
        current.get(
            "contact",
            "세부 공고 확인"
        )
    )


    st.divider()


    # =====================================================
    # 근거자료
    # =====================================================

    with st.expander(
        "근거자료 확인"
    ):

        st.write(
            current.get(
                "source",
                "출처 확인 필요"
            )
        )


        st.write(
            f"{current.get('page', '확인 필요')}페이지"
        )


    # =====================================================
    # 페이지 이동
    # =====================================================

    col1, col2, col3 = st.columns(
        [1, 1.3, 1]
    )


    with col1:

        if st.button(
            "← 이전",
            use_container_width=True,
            disabled=(index == 0),
        ):

            st.session_state.result_index -= 1

            st.rerun()


    with col2:

        if st.button(
            "처음으로",
            use_container_width=True,
        ):

            go_home()


    with col3:

        if st.button(
            "다음 →",
            use_container_width=True,
            disabled=(
                index >= total - 1
            ),
        ):

            st.session_state.result_index += 1

            st.rerun()


    st.caption(
        "※ 실제 신청 전에는 담당기관의 최신 공고를 "
        "한 번 더 확인해주세요."
    )