import os
import re
import json
from datetime import date
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

VECTOR_SEARCH_COUNT = 90
FINAL_CONTEXT_COUNT = 16
MAX_RESULTS = 3

# 강원신용보증재단 운영상품 PDF
SINBO_SOURCE = "강원신용보증재단_2026_운영보증상품.pdf"

# 강원신보 자료를 '우대'하지는 않지만
# 보증/융자 검색 시 후보자료에서 아예 빠지는 현상을 방지하기 위한 수
SINBO_CONTEXT_MAX = 4


st.set_page_config(
    page_title="강원 소상공인 혜택 도우미",
    page_icon="💼",
    layout="centered",
)


# =========================================================
# 2. HTML 출력
# =========================================================

def render_html(markup):

    cleaned = "\n".join(
        line.strip()
        for line in str(markup).splitlines()
        if line.strip()
    )

    st.markdown(
        cleaned,
        unsafe_allow_html=True,
    )


def html_escape(text):

    text = str(text or "")

    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# =========================================================
# 3. 금액을 읽기 쉽게 변환
#
# 예:
# 533천 원     → 53만 3천 원
# 6,600천 원   → 660만 원
# 70,000천 원  → 7,000만 원
# =========================================================

def korean_won_from_thousand(value):

    try:
        thousand_units = int(
            round(float(value))
        )
    except Exception:
        return None

    won = thousand_units * 1000

    man = won // 10000
    remainder = won % 10000
    cheon = remainder // 1000

    parts = []

    if man > 0:
        parts.append(
            f"{man:,}만"
        )

    if cheon > 0:
        parts.append(
            f"{cheon}천"
        )

    if not parts:
        return f"{thousand_units:,}천 원"

    return " ".join(parts) + " 원"


def normalize_amount_display(text):

    text = str(
        text or ""
    )

    pattern = re.compile(
        r"(?<![\d.])"
        r"(\d[\d,]*(?:\.\d+)?)"
        r"\s*천\s*원"
    )

    def replace_match(match):

        raw_number = (
            match.group(1)
            .replace(",", "")
        )

        converted = korean_won_from_thousand(
            raw_number
        )

        if converted:
            return converted

        return match.group(0)

    return pattern.sub(
        replace_match,
        text
    )


# =========================================================
# 4. 연락처 정규화
#
# 강원신용보증재단 자료 중
# 033-260-0000~1 같은 범위 표기를
# Gemini가 033-260-0000-1로 읽는 경우를 교정
# =========================================================

def normalize_contact_display(contact):

    text = str(
        contact or ""
    )

    # 033-260-0000~1
    # 033-260-0000～1
    # 033-260-0000∼1
    # 033-260-0000-1
    # 등을 모두 033-260-0001로 통일
    text = re.sub(
        r"033\s*[-)]?\s*260\s*-\s*0000"
        r"\s*(?:~|～|∼|-)\s*1",
        "033-260-0001",
        text,
    )

    # 괄호형 표현
    text = re.sub(
        r"033\s*\)\s*260\s*-\s*0001",
        "033-260-0001",
        text,
    )

    # 공백이 섞인 정상번호도 통일
    text = re.sub(
        r"033\s*-\s*260\s*-\s*0001",
        "033-260-0001",
        text,
    )

    return text


# =========================================================
# 5. 디자인
# =========================================================

render_html(
    """
<style>

:root {
    --navy: #123A59;
    --teal: #277B77;
    --text: #294457;
    --muted: #718493;
    --line: #D9E4E9;

    --green-bg: #E9F7F0;
    --green: #17694E;

    --yellow-bg: #FFF4D7;
    --yellow: #875A00;

    --red-bg: #FCEBEC;
    --red: #9D3030;
}


html,
body,
[class*="css"] {
    font-family:
        "Pretendard",
        "Noto Sans KR",
        "Apple SD Gothic Neo",
        sans-serif;
}


.stApp {
    background:
        radial-gradient(
            circle at 95% 2%,
            rgba(39,123,119,0.10),
            transparent 26%
        ),
        linear-gradient(
            180deg,
            #F2F7F8 0%,
            #F8FAFB 52%,
            #F3F6F8 100%
        );
}


.block-container {
    max-width: 790px !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}


h1 {
    color: var(--navy) !important;
    font-size: 2.55rem !important;
    line-height: 1.18 !important;
    font-weight: 900 !important;
    letter-spacing: -0.05em !important;
    margin-bottom: 0.75rem !important;
}


h3 {
    color: #173E5A !important;
    font-size: 1.43rem !important;
    font-weight: 850 !important;
    letter-spacing: -0.035em !important;
}


p,
li {
    color: var(--text);
    font-size: 1.10rem;
    line-height: 1.72;
}


[data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
    font-size: 1.02rem !important;
    line-height: 1.6 !important;
}


/* =========================================================
   첫 화면
========================================================= */

.hero-card {
    background: rgba(255,255,255,0.97);
    border: 1px solid #D7E3E9;
    border-radius: 24px;
    padding: 1.45rem 1.5rem;
    margin: 0.7rem 0 1.65rem 0;
    box-shadow: 0 15px 38px rgba(30,58,78,0.08);
}


.hero-label {
    display: inline-block;
    background: #E7F4F2;
    color: #246B67;
    border-radius: 999px;
    padding: 0.4rem 0.74rem;
    font-size: 0.99rem;
    font-weight: 850;
    margin-bottom: 0.82rem;
}


.hero-title {
    color: var(--navy);
    font-size: 1.68rem;
    line-height: 1.4;
    font-weight: 900;
    letter-spacing: -0.04em;
    margin-bottom: 0.55rem;
}


.hero-text {
    color: #536C7E;
    font-size: 1.12rem;
    line-height: 1.72;
    font-weight: 560;
}


/* =========================================================
   입력창
========================================================= */

.stTextArea textarea {
    min-height: 155px !important;
    background: #FFFFFF !important;
    color: #203B50 !important;
    border: 2px solid #D4E0E6 !important;
    border-radius: 18px !important;
    padding: 1.1rem 1.15rem !important;
    font-size: 1.22rem !important;
    line-height: 1.7 !important;

    box-shadow:
        0 8px 22px rgba(29,57,76,0.06)
        !important;
}


.stTextArea textarea:focus {
    border-color: #2E827D !important;

    box-shadow:
        0 0 0 4px rgba(46,130,125,0.12),
        0 10px 26px rgba(29,57,76,0.08)
        !important;
}


.stTextArea textarea::placeholder {
    color: #879AA7 !important;
    opacity: 1 !important;
}


/* =========================================================
   버튼
========================================================= */

.stButton > button {
    min-height: 64px !important;
    border-radius: 15px !important;
    border: 1px solid #D3E0E6 !important;
    background: #FFFFFF !important;
    color: #23465D !important;
    font-size: 1.13rem !important;
    font-weight: 850 !important;
    letter-spacing: -0.025em !important;
}


.stButton > button:hover {
    border-color: #2E827D !important;
    color: #176B66 !important;
    background: #EFF8F7 !important;

    box-shadow:
        0 8px 18px rgba(39,123,119,0.11);
}


button[data-testid="baseButton-primary"] {
    min-height: 72px !important;
    border: none !important;
    border-radius: 17px !important;

    background:
        linear-gradient(
            135deg,
            #123A59 0%,
            #205D73 58%,
            #2D807A 100%
        )
        !important;

    color: #FFFFFF !important;
    font-size: 1.25rem !important;
    font-weight: 900 !important;

    box-shadow:
        0 12px 28px rgba(18,58,89,0.22)
        !important;
}


button[data-testid="baseButton-primary"] *,
button[data-testid="baseButton-primary"] p,
button[data-testid="baseButton-primary"] span,
button[data-testid="baseButton-primary"] div {
    color: #FFFFFF !important;
}


[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #DCE6EB !important;
    border-radius: 15px !important;
    overflow: hidden !important;
}


[data-testid="stExpander"] summary {
    color: #38566B !important;
    font-size: 1.06rem !important;
    font-weight: 800 !important;
}


hr {
    border: 0 !important;
    border-top: 1px solid #DDE6EB !important;
    margin-top: 1.55rem !important;
    margin-bottom: 1.55rem !important;
}


/* =========================================================
   결과 상단
========================================================= */

.result-card {
    background: rgba(255,255,255,0.98);
    border: 1px solid #D9E5EA;
    border-radius: 24px;
    padding: 1.45rem;
    margin: 0.8rem 0 1rem 0;

    box-shadow:
        0 16px 40px rgba(28,55,75,0.09);
}


.amount-label {
    color: #647C8C;
    font-size: 1.02rem;
    font-weight: 800;
    margin-bottom: 0.16rem;
}


.amount-main {
    color: #103B5B;
    font-size: 2.35rem;
    line-height: 1.18;
    font-weight: 950;
    letter-spacing: -0.055em;
    margin-bottom: 0.7rem;
}


.policy-name {
    color: #173F5B;
    font-size: 1.6rem;
    line-height: 1.4;
    font-weight: 900;
    letter-spacing: -0.04em;
    margin-bottom: 0.75rem;
}


.type-pill {
    display: inline-block;
    background: #EDF3F6;
    color: #5B7181;
    border-radius: 999px;
    padding: 0.42rem 0.72rem;
    font-size: 0.98rem;
    font-weight: 800;
}


/* =========================================================
   공통 카드
========================================================= */

.info-card {
    background: #FFFFFF;
    border: 1px solid #DFE8ED;
    border-radius: 18px;
    padding: 1.1rem 1.15rem;
    margin: 0.75rem 0;

    box-shadow:
        0 5px 15px rgba(30,56,76,0.035);
}


.info-title {
    color: #23475F;
    font-size: 1.14rem;
    font-weight: 900;
    margin-bottom: 0.55rem;
}


/* =========================================================
   내 조건 확인
========================================================= */

.condition-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding: 0.78rem 0;

    border-bottom:
        1px solid #E7EEF2;
}


.condition-row:last-child {
    border-bottom: none;
}


.condition-label {
    color: #38566A;
    font-size: 1.12rem;
    font-weight: 850;
}


.condition-status {
    border-radius: 999px;
    padding: 0.48rem 0.82rem;
    font-size: 1.04rem;
    font-weight: 900;
    white-space: nowrap;
}


.status-good {
    background: var(--green-bg);
    color: var(--green);
}


.status-warn {
    background: var(--yellow-bg);
    color: var(--yellow);
}


.status-bad {
    background: var(--red-bg);
    color: var(--red);
}


/* =========================================================
   신청상태
========================================================= */

.application-card {
    background:
        linear-gradient(
            135deg,
            #FFFFFF,
            #F7FAFB
        );

    border:
        1px solid #DBE6EB;
    border-radius: 18px;
    padding: 1.1rem 1.15rem;
    margin: 0.75rem 0;
}


.application-status {
    display: inline-block;
    border-radius: 999px;
    padding: 0.48rem 0.82rem;
    font-size: 1.10rem;
    font-weight: 900;
    margin-bottom: 0.5rem;
}


.application-detail {
    color: #526B7D;
    font-size: 1.08rem;
    line-height: 1.6;
    font-weight: 570;
}


/* =========================================================
   전화
========================================================= */

.phone-card {
    background:
        linear-gradient(
            135deg,
            #EDF7F5,
            #F6FAFB
        );

    border:
        1px solid #D0E5E2;
    border-radius: 18px;
    padding: 1.1rem 1.15rem;
    margin: 0.85rem 0 0.65rem 0;
}


.phone-label {
    color: #22615D;
    font-size: 1.05rem;
    font-weight: 850;
    margin-bottom: 0.25rem;
}


.phone-main {
    color: #183F58;
    font-size: 1.30rem;
    line-height: 1.5;
    font-weight: 900;
}


.tel-button {
    display: block;
    width: 100%;
    box-sizing: border-box;

    padding: 1.05rem;
    margin: 0.55rem 0 0.8rem 0;

    text-align: center;
    text-decoration: none !important;

    color: #FFFFFF !important;

    background:
        linear-gradient(
            135deg,
            #277B76,
            #22627B
        );

    border-radius: 15px;
    font-size: 1.20rem;
    font-weight: 900;

    box-shadow:
        0 9px 22px rgba(34,98,123,0.18);
}


/* =========================================================
   전화 질문
========================================================= */

.call-card {
    background: #FFFDF7;
    border: 1px solid #EEE0B8;
    border-radius: 18px;
    padding: 1.1rem 1.15rem;
    margin: 0.8rem 0;
}


.call-title {
    color: #72571B;
    font-size: 1.14rem;
    font-weight: 900;
    margin-bottom: 0.65rem;
}


.call-line {
    color: #434C55;
    font-size: 1.12rem;
    line-height: 1.72;
    font-weight: 580;
    margin: 0.38rem 0;
}


/* =========================================================
   모바일
========================================================= */

@media (max-width: 640px) {

    .block-container {
        padding-top: 1.15rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
        padding-bottom: 2.8rem !important;
    }

    h1 {
        font-size: 2.05rem !important;
    }

    h3 {
        font-size: 1.3rem !important;
    }

    .hero-card {
        padding: 1.15rem;
        border-radius: 20px;
    }

    .hero-title {
        font-size: 1.46rem;
    }

    .hero-text {
        font-size: 1.08rem;
    }

    .stTextArea textarea {
        min-height: 165px !important;
        font-size: 1.17rem !important;
    }

    .stButton > button {
        min-height: 65px !important;
        font-size: 1.10rem !important;
    }

    button[data-testid="baseButton-primary"] {
        min-height: 73px !important;
        font-size: 1.19rem !important;
    }

    .amount-main {
        font-size: 2.05rem;
    }

    .policy-name {
        font-size: 1.42rem;
    }

    .condition-label {
        font-size: 1.07rem;
    }

    .condition-status {
        font-size: 1rem;
    }
}

</style>
"""
)


# =========================================================
# 6. Session State
# =========================================================

DEFAULT_STATE = {
    "screen": "search",
    "results": [],
    "result_index": 0,
    "original_question": "",
    "clarify_question": "",
    "selected_support": "전체",
    "call_scripts": {},
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 7. Gemini
# =========================================================

API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


if not API_KEY:

    st.error(
        "GEMINI_API_KEY가 설정되어 있지 않습니다."
    )

    st.stop()


gemini_client = genai.Client(
    api_key=API_KEY
)


# =========================================================
# 8. Embedding / ChromaDB
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


embedding_model = load_embedding_model()


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

    st.code(
        str(e)
    )

    st.stop()


@st.cache_resource
def load_all_records():

    data = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    return (
        data.get(
            "documents",
            []
        ),
        data.get(
            "metadatas",
            []
        ),
    )


all_documents, all_metadatas = (
    load_all_records()
)


# =========================================================
# 9. 페이지 인덱스
# =========================================================

@st.cache_resource
def build_page_index():

    index = defaultdict(list)

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

        try:
            page = int(page)

        except Exception:
            continue

        index[
            (
                source,
                page
            )
        ].append(
            document
        )

    return index


page_index = build_page_index()


# =========================================================
# 10. 공통 문자열
# =========================================================

def normalize_text(text):

    if text is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text).lower()
    ).strip()


def clean_json_text(text):

    text = str(
        text or ""
    ).strip()

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
# 11. 지역
# =========================================================

GANGWON_CITIES = [
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
]


OTHER_REGIONS = [
    "서울",
    "경기",
    "경기도",
    "인천",
    "충북",
    "충청북도",
    "충남",
    "충청남도",
    "대전",
    "세종",
    "전북",
    "전북특별자치도",
    "전남",
    "전라남도",
    "광주",
    "경북",
    "경상북도",
    "경남",
    "경상남도",
    "대구",
    "부산",
    "울산",
    "제주",
]


def detect_city(question):

    q = normalize_text(
        question
    )

    for city in GANGWON_CITIES:

        if city in q:
            return city

    return None


def detect_region(question):

    q = normalize_text(
        question
    )

    if (
        "강원" in q
        or detect_city(q)
    ):
        return "강원"

    for region in OTHER_REGIONS:

        if region in q:
            return region

    return None


# =========================================================
# 12. 지원 분야
# =========================================================

INTENT_GROUPS = {

    "창업": [
        "창업",
        "예비창업",
        "초기창업",
        "스타트업",
        "사업을 시작",
        "가게를 열",
    ],

    "소상공인": [
        "소상공인",
        "자영업",
        "가게",
        "식당",
        "음식점",
        "카페",
        "매장",
    ],

    "고용": [
        "고용",
        "채용",
        "직원",
        "근로자",
        "인건비",
        "장려금",
    ],

    "마케팅": [
        "마케팅",
        "홍보",
        "광고",
        "판로",
        "온라인 판매",
        "쇼핑몰",
        "브랜딩",
    ],

    "시설": [
        "시설",
        "설비",
        "장비",
        "기계",
        "리모델링",
        "간판",
        "키오스크",
    ],

    "기술": [
        "ai",
        "인공지능",
        "기술개발",
        "연구개발",
        "r&d",
        "스마트공장",
        "자동화",
    ],

    "수출": [
        "수출",
        "해외진출",
        "글로벌",
        "해외시장",
    ],
}


def detect_intents(question):

    q = normalize_text(
        question
    )

    result = []

    for intent, terms in INTENT_GROUPS.items():

        if any(
            term in q
            for term in terms
        ):
            result.append(intent)

    return result


# =========================================================
# 13. 지원 유형
# =========================================================

SUPPORT_TYPE_TERMS = {

    "지원금": [
        "지원금",
        "보조금",
        "장려금",
        "인건비",
        "바우처",
        "사업비",
    ],

    "융자": [
        "대출",
        "융자",
        "정책자금",
        "운영자금",
        "운전자금",
        "시설자금",
        "금리",
    ],

    "보증": [
        "보증",
        "신용보증",
        "보증서",
        "담보가 부족",
        "담보 부족",
    ],

    "R&D": [
        "r&d",
        "연구개발",
        "기술개발",
        "연구비",
    ],
}


def detect_support_type(question):

    q = normalize_text(
        question
    )

    found = []

    for support_type, terms in SUPPORT_TYPE_TERMS.items():

        if any(
            term in q
            for term in terms
        ):
            found.append(
                support_type
            )

    if len(found) == 1:
        return found[0]

    return "전체"


# =========================================================
# 14. 키워드
# =========================================================

def extract_keywords(question):

    words = re.findall(
        r"[가-힣A-Za-z0-9&]+",
        normalize_text(question)
    )

    stopwords = {
        "무엇",
        "어떤",
        "있나요",
        "알려주세요",
        "받을",
        "있는",
        "대한",
        "관련",
        "지원",
        "지원사업",
        "사업",
        "정책",
        "가능",
        "도움",
        "혜택",
        "필요",
    }

    result = []

    for word in words:

        if len(word) < 2:
            continue

        if word in stopwords:
            continue

        if word not in result:
            result.append(word)

    return result


# =========================================================
# 15. Hard Filter
# =========================================================

EXCLUSIVE_MARKERS = [
    "지원대상",
    "신청대상",
    "대상기업",
    "관내",
    "소재기업",
    "사업장 소재",
    "본사 소재",
]


def hard_exclusion_reason(
    document,
    source,
    question
):

    text = normalize_text(
        f"{source} {document}"
    )

    q = normalize_text(
        question
    )

    user_region = detect_region(
        question
    )

    user_city = detect_city(
        question
    )

    if (
        user_region
        and user_region != "강원"
    ):
        return "강원 외 지역"

    exclusive = any(
        marker in text
        for marker in EXCLUSIVE_MARKERS
    )

    if (
        user_region == "강원"
        and exclusive
    ):

        other_regions = [
            region
            for region in OTHER_REGIONS
            if region in text
        ]

        if (
            other_regions
            and "강원" not in text
        ):
            return "다른 지역 전용"

    if (
        user_city
        and exclusive
    ):

        other_cities = [
            city
            for city in GANGWON_CITIES
            if (
                city != user_city
                and city in text
            )
        ]

        broad_terms = [
            "전국",
            "강원특별자치도",
            "강원도",
            "도내",
            "중소벤처기업부",
            "고용노동부",
            "소상공인시장진흥공단",
        ]

        if (
            other_cities
            and user_city not in text
            and not any(
                term in text
                for term in broad_terms
            )
        ):
            return "다른 시군 전용"

    special_groups = [
        [
            "육아휴직 대체인력",
            "출산육아기",
        ],
        [
            "장애인 고용장려금",
            "장애인고용",
        ],
        [
            "외국인력 지원",
            "외국인 고용허가",
        ],
        [
            "계속고용장려금",
            "정년연장",
        ],
    ]

    for group in special_groups:

        if (
            any(
                term in text
                for term in group
            )
            and not any(
                term in q
                for term in group
            )
        ):
            return "특수자격 전용"

    return None


# =========================================================
# 16. Vector Search
# =========================================================

def vector_search(question):

    embedding = (
        embedding_model
        .encode(
            question,
            normalize_embeddings=True,
        )
        .tolist()
    )

    results = collection.query(
        query_embeddings=[
            embedding
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
# 17. 후보 점수
#
# 주의:
# 강원신용보증재단이라는 이유로 별도 가산점을 주지 않습니다.
# =========================================================

def score_candidate(
    document,
    source,
    question,
    distance
):

    q = normalize_text(
        question
    )

    text = normalize_text(
        f"{source} {document}"
    )

    score = 0.0

    if distance is not None:

        score += max(
            0.0,
            1.0 - distance
        ) * 10.0

    keywords = extract_keywords(
        question
    )

    score += (
        sum(
            1
            for keyword in keywords
            if keyword in text
        )
        * 1.5
    )

    intents = detect_intents(
        question
    )

    for intent in intents:

        hits = sum(
            1
            for term in INTENT_GROUPS.get(
                intent,
                []
            )
            if term in text
        )

        if hits:
            score += (
                3.0
                + min(
                    hits,
                    3
                ) * 0.7
            )

    city = detect_city(
        question
    )

    if city:

        if city in text:
            score += 6.0

        elif any(
            term in text
            for term in [
                "강원특별자치도",
                "강원도",
                "도내",
                "전국",
            ]
        ):
            score += 1.5

    if (
        "강원" in q
        and "강원" in text
    ):
        score += 4.0

    if any(
        term in text
        for term in [
            "지원금",
            "보조금",
            "장려금",
            "인건비",
            "지원한도",
            "대출",
            "융자",
            "보증한도",
            "만원",
            "억원",
        ]
    ):
        score += 2.0

    return score


# =========================================================
# 18. 후보 선정
#
# 핵심 변경:
# 기존 상위 후보는 그대로 유지합니다.
#
# 단, 보증/융자 관련 질문일 경우
# 강원신용보증재단 운영상품 자료가 검색 후보에서
# 완전히 빠지는 경우만 방지합니다.
#
# 이것은 추천 순위 가산점이 아닙니다.
# Gemini가 비교할 자료를 볼 기회만 보장합니다.
# =========================================================

def build_ranked_candidates(
    question,
    support_type
):

    query = question

    if support_type == "지원금":

        query += (
            " 지원금 보조금 장려금 "
            "인건비 사업비 바우처"
        )

    elif support_type == "융자":

        query += (
            " 대출 융자 정책자금 "
            "운전자금 경영안정자금"
        )

    elif support_type == "보증":

        query += (
            " 신용보증 협약보증 보증서"
        )

    elif support_type == "R&D":

        query += (
            " 연구개발 기술개발 R&D 연구비"
        )

    vector_results = vector_search(
        query
    )

    distance_map = {
        normalize_text(
            item["document"]
        ):
        item["distance"]

        for item in vector_results
    }

    candidates = []
    seen = set()

    for document, metadata in zip(
        all_documents,
        all_metadatas
    ):

        if not document:
            continue

        normalized = normalize_text(
            document
        )

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        metadata = metadata or {}

        source = metadata.get(
            "source",
            "출처 미상"
        )

        if hard_exclusion_reason(
            document,
            source,
            question
        ):
            continue

        score = score_candidate(
            document,
            source,
            question,
            distance_map.get(
                normalized
            ),
        )

        text = normalize_text(
            document
        )

        if support_type == "지원금":

            if any(
                term in text
                for term in [
                    "지원금",
                    "보조금",
                    "장려금",
                    "인건비",
                    "바우처",
                ]
            ):
                score += 4.0

        elif support_type == "융자":

            if any(
                term in text
                for term in [
                    "융자",
                    "대출",
                    "정책자금",
                    "운전자금",
                ]
            ):
                score += 4.0

        elif support_type == "보증":

            if any(
                term in text
                for term in [
                    "보증",
                    "신용보증",
                    "보증한도",
                    "협약보증",
                    "특례보증",
                ]
            ):
                score += 4.0

        elif support_type == "R&D":

            if any(
                term in text
                for term in [
                    "r&d",
                    "연구개발",
                    "기술개발",
                ]
            ):
                score += 4.0

        if score > 0:

            candidates.append(
                {
                    "document": document,
                    "metadata": metadata,
                    "score": score,
                }
            )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    selected = []
    page_seen = set()
    source_counts = defaultdict(int)

    # -----------------------------------------------------
    # 기존 방식 그대로 상위 후보 선정
    # -----------------------------------------------------

    for item in candidates:

        metadata = item["metadata"]

        source = metadata.get(
            "source",
            "출처 미상"
        )

        page = metadata.get(
            "page",
            "페이지 미상"
        )

        key = (
            source,
            page
        )

        if key in page_seen:
            continue

        if source_counts[source] >= 5:
            continue

        page_seen.add(key)
        source_counts[source] += 1

        selected.append(item)

        if len(selected) >= FINAL_CONTEXT_COUNT:
            break

    # -----------------------------------------------------
    # 강원신보 자료 후보 다양성 보장
    #
    # 중요:
    # - 기존 후보를 삭제하지 않음
    # - 점수를 올리지 않음
    # - 추천 결과에 반드시 넣는 것도 아님
    # - Gemini가 비교할 후보자료에만 추가
    # -----------------------------------------------------

    q = normalize_text(
        question
    )

    guarantee_or_loan_query = (
        support_type in [
            "보증",
            "융자",
            "전체",
        ]
        or any(
            term in q
            for term in [
                "보증",
                "신용보증",
                "보증서",
                "대출",
                "융자",
                "자금",
                "담보",
            ]
        )
    )

    if guarantee_or_loan_query:

        sinbo_candidates = [
            item
            for item in candidates
            if item["metadata"].get(
                "source",
                ""
            ) == SINBO_SOURCE
        ]

        added_count = 0

        for item in sinbo_candidates:

            metadata = item[
                "metadata"
            ]

            source = metadata.get(
                "source",
                ""
            )

            page = metadata.get(
                "page",
                "페이지 미상"
            )

            key = (
                source,
                page
            )

            if key in page_seen:
                continue

            selected.append(
                item
            )

            page_seen.add(
                key
            )

            added_count += 1

            if (
                added_count
                >= SINBO_CONTEXT_MAX
            ):
                break

    return selected


# =========================================================
# 19. 페이지 확장
# =========================================================

def get_expanded_context(
    source,
    page
):

    try:
        page = int(page)

    except Exception:
        return ""

    output = []

    for target_page in [
        page - 1,
        page,
        page + 1,
    ]:

        if target_page < 1:
            continue

        chunks = page_index.get(
            (
                source,
                target_page
            ),
            []
        )

        if chunks:

            output.append(
                f"""
[페이지 {target_page}]

{" ".join(chunks)}
"""
            )

    return "\n".join(
        output
    )


# =========================================================
# 20. 날짜 및 신청상태
# =========================================================

def extract_full_dates(text):

    text = str(
        text or ""
    )

    found = []

    pattern = (
        r"(20\d{2})\s*[.\-/년]\s*"
        r"(\d{1,2})\s*[.\-/월]\s*"
        r"(\d{1,2})"
    )

    for match in re.finditer(
        pattern,
        text
    ):

        try:

            found.append(
                date(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                )
            )

        except Exception:
            pass

    return found


def extract_year_months(text):

    text = str(
        text or ""
    )

    found = []

    for match in re.finditer(
        r"(20\d{2})\s*[.\-/년]\s*(\d{1,2})\s*월",
        text
    ):

        try:

            year = int(
                match.group(1)
            )

            month = int(
                match.group(2)
            )

            if 1 <= month <= 12:

                pair = (
                    year,
                    month
                )

                if pair not in found:
                    found.append(pair)

        except Exception:
            pass

    year_match = re.search(
        r"(20\d{2})",
        text
    )

    if year_match:

        default_year = int(
            year_match.group(1)
        )

        for match in re.finditer(
            r"(?<!\d)(\d{1,2})\s*월",
            text
        ):

            month = int(
                match.group(1)
            )

            pair = (
                default_year,
                month
            )

            if (
                1 <= month <= 12
                and pair not in found
            ):
                found.append(pair)

    return found


def determine_application_status(
    period_text
):

    text = str(
        period_text or ""
    ).strip()

    normalized = normalize_text(
        text
    )

    today = date.today()

    if (
        not text
        or "확인 필요" in normalized
    ):

        return {
            "status":
                "확인 필요",

            "detail":
                "담당기관에 현재 접수 여부를 확인해주세요.",

            "exclude":
                False,
        }

    if any(
        term in normalized
        for term in [
            "상시 접수",
            "상시접수",
            "연중 접수",
            "연중접수",
            "상시 신청",
            "상시신청",
        ]
    ):

        return {
            "status":
                "상시 접수",

            "detail":
                "상시 접수로 안내된 사업입니다.",

            "exclude":
                False,
        }

    if any(
        term in normalized
        for term in [
            "예산 소진",
            "자금 소진",
            "소진 시까지",
            "소진 시 까지",
            "한도 소진",
            "한도소진",
        ]
    ):

        return {
            "status":
                "확인 필요",

            "detail":
                "현재 한도나 예산이 남아 있는지 담당기관에 확인해주세요.",

            "exclude":
                False,
        }

    full_dates = extract_full_dates(
        text
    )

    if len(full_dates) >= 2:

        start = full_dates[0]
        end = full_dates[-1]

        if today < start:

            return {
                "status":
                    "접수 예정",

                "detail":
                    f"{start.month}월 {start.day}일부터 "
                    f"접수 예정입니다.",

                "exclude":
                    False,
            }

        if start <= today <= end:

            return {
                "status":
                    "접수 중",

                "detail":
                    f"{end.month}월 {end.day}일까지입니다.",

                "exclude":
                    False,
            }

        return {
            "status":
                "접수 종료",

            "detail":
                f"{end.month}월 {end.day}일에 "
                f"마감되었습니다.",

            "exclude":
                True,
        }

    if (
        len(full_dates) == 1
        and any(
            term in normalized
            for term in [
                "까지",
                "마감",
                "접수 종료",
                "신청 종료",
            ]
        )
    ):

        end = full_dates[0]

        if today <= end:

            return {
                "status":
                    "접수 중",

                "detail":
                    f"{end.month}월 {end.day}일까지입니다.",

                "exclude":
                    False,
            }

        return {
            "status":
                "접수 종료",

            "detail":
                f"{end.month}월 {end.day}일에 "
                f"마감되었습니다.",

            "exclude":
                True,
        }

    year_months = extract_year_months(
        text
    )

    if (
        year_months
        and any(
            term in normalized
            for term in [
                "접수 예정",
                "접수예정",
                "신청 예정",
                "신청예정",
            ]
        )
    ):

        target_year, target_month = (
            year_months[-1]
        )

        current_pair = (
            today.year,
            today.month
        )

        target_pair = (
            target_year,
            target_month
        )

        if current_pair < target_pair:

            return {
                "status":
                    "접수 예정",

                "detail":
                    f"{target_year}년 "
                    f"{target_month}월 접수 예정입니다.",

                "exclude":
                    False,
            }

        if current_pair == target_pair:

            return {
                "status":
                    "확인 필요",

                "detail":
                    "예정된 접수 시기입니다. "
                    "현재 접수 여부를 확인해주세요.",

                "exclude":
                    False,
            }

        return {
            "status":
                "기간 경과",

            "detail":
                "예정된 접수 시기가 이미 지났습니다.",

            "exclude":
                True,
        }

    if year_months:

        target_year, target_month = (
            year_months[-1]
        )

        current_pair = (
            today.year,
            today.month
        )

        target_pair = (
            target_year,
            target_month
        )

        if current_pair > target_pair:

            return {
                "status":
                    "기간 경과",

                "detail":
                    "안내된 일정이 이미 지난 사업입니다.",

                "exclude":
                    True,
            }

    return {
        "status":
            "확인 필요",

        "detail":
            "담당기관에 현재 접수 여부를 확인해주세요.",

        "exclude":
            False,
    }


# =========================================================
# 21. 전화번호
# =========================================================

PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:0\d{1,2}[-)]?\s*\d{3,4}-\d{4}"
    r"|1[2-9]\d{2})"
    r"(?!\d)"
)


def extract_phone(text):

    if not text:
        return None

    # 먼저 잘못된 표기를 교정
    text = normalize_contact_display(
        text
    )

    match = PHONE_PATTERN.search(
        str(text)
    )

    if not match:
        return None

    return (
        match.group(0)
        .replace(")", "-")
        .replace(" ", "")
    )


# =========================================================
# 22. 결과 검증
# =========================================================

def validate_result(
    result,
    candidates
):

    sources = {
        item["metadata"].get(
            "source",
            "출처 미상"
        )
        for item in candidates
    }

    source = result.get(
        "source",
        ""
    )

    if source not in sources:
        return None

    result[
        "application_period"
    ] = (
        result.get(
            "application_period"
        )
        or "신청기간 확인 필요"
    )

    status_info = (
        determine_application_status(
            result[
                "application_period"
            ]
        )
    )

    if status_info.get(
        "exclude",
        False
    ):
        return None

    result[
        "application_status"
    ] = status_info[
        "status"
    ]

    result[
        "application_detail"
    ] = status_info[
        "detail"
    ]

    eligibility = result.get(
        "eligibility",
        []
    )

    if not isinstance(
        eligibility,
        list
    ):
        eligibility = []

    cleaned = []

    for item in eligibility[:4]:

        if not isinstance(
            item,
            dict
        ):
            continue

        status = item.get(
            "status",
            "확인 필요"
        )

        if status not in [
            "적합",
            "확인 필요",
            "부적합",
        ]:
            status = "확인 필요"

        cleaned.append(
            {
                "label":
                    item.get(
                        "label",
                        "조건"
                    ),

                "status":
                    status,
            }
        )

    result[
        "eligibility"
    ] = cleaned

    if any(
        item["status"] == "부적합"
        for item in cleaned
    ):
        return None

    # -----------------------------------------------------
    # 연락처 정규화
    # -----------------------------------------------------

    contact = normalize_contact_display(
        result.get(
            "contact",
            ""
        )
    )

    result[
        "contact"
    ] = contact

    phone = extract_phone(
        contact
    )

    if phone:

        context = get_expanded_context(
            source,
            result.get(
                "page"
            )
        )

        # 원본 자체에 033-260-0000~1이 있는 경우도
        # 같은 방식으로 정규화한 뒤 비교
        normalized_context = (
            normalize_contact_display(
                context
            )
        )

        phone_digits = re.sub(
            r"\D",
            "",
            phone
        )

        context_digits = re.sub(
            r"\D",
            "",
            normalized_context
        )

        if (
            phone_digits
            and phone_digits not in context_digits
        ):

            result[
                "contact"
            ] = (
                "세부 공고 확인"
            )

    return result


# =========================================================
# 23. Gemini 결과 생성
# =========================================================

def generate_results(
    question,
    support_type
):

    candidates = (
        build_ranked_candidates(
            question,
            support_type
        )
    )

    if not candidates:
        return []

    contexts = []

    for number, item in enumerate(
        candidates,
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

        contexts.append(
            f"""
========================================
후보자료 {number}
========================================

파일:
{source}

검색 중심 페이지:
{page}

{get_expanded_context(
    source,
    page
)}
"""
        )

    context = "\n\n".join(
        contexts
    )

    support_rules = {

        "지원금":
            "지원금·보조금·장려금·인건비·바우처 등 "
            "비용지원 성격 사업을 우선 선정하세요.",

        "융자":
            "융자·대출·정책자금 사업을 우선 선정하세요.",

        "보증":
            "신용보증·협약보증 등 보증사업을 우선 선정하세요.",

        "R&D":
            "연구개발비·기술개발비 지원사업을 우선 선정하세요.",

        "전체":
            "사용자 상황에 직접 도움이 되는 사업을 "
            "종류에 관계없이 선정하세요.",
    }

    prompt = f"""
당신은 디지털 정보 접근에 어려움을 겪을 수 있는
강원지역 소상공인을 위한 금융·정책지원 안내 전문가입니다.

반드시 아래 정책자료만 이용하세요.

사용자의 상황:
{question}

검색 지원유형:
{support_type}

지원유형 규칙:
{support_rules.get(
    support_type,
    support_rules["전체"]
)}

현재 날짜는 {date.today().isoformat()} 입니다.

==================================================
가장 중요한 추천 순서
==================================================

추천 결과의 순서는 반드시 다음 원칙을 따르세요.

1순위:
사용자가 필요로 하는 지원종류와 얼마나 직접적으로 맞는지

2순위:
같은 지원종류의 사업이라면
실제로 한 기업 또는 한 사업자가 받을 수 있는
지원금액·대출한도·보증한도가 큰 사업

3순위:
사용자의 지역·업종·사업상황·자격조건과의 적합성

중요:
강원신용보증재단 자료라는 이유만으로
추천 순위를 높이지 마세요.

춘천시·원주시·정선군 등 지역 전용 사업이라는 이유만으로
전국 또는 강원도 전체 사업보다 먼저 추천하지 마세요.

포괄적인 안내자료와 개별상품을 모두 공정하게 비교하고,
사용자가 실제로 받을 수 있는 혜택과 지원종류를 중심으로
최종 순서를 결정하세요.

사업 전체의 총 사업비나 총 공급규모는
사용자 개인의 지원금액으로 비교하면 안 됩니다.

예:
"지원규모 100억 원"은 개인이 받는 금액이 아닙니다.
"업체당 최대 5천만 원"을 실제 혜택으로 판단해야 합니다.

==================================================
반드시 지켜야 할 규칙
==================================================

1. 사용자의 지역, 업종, 필요와
직접 관련된 사업만 선정하세요.

2. 최대 {MAX_RESULTS}개까지만 선정하세요.

3. 적합한 사업이 적으면
억지로 3개를 채우지 마세요.

4. 다른 지역 전용사업은 제외하세요.

5. 사용자가 말하지 않은 특별한 자격조건은
임의로 가정하지 마세요.

6. 동일한 사업은 한 번만 추천하세요.

7. 변경공고와 수정공고가 있으면
최신 공고 내용을 우선하세요.

8. 지원금액, 대출한도, 보증한도,
지원대상, 신청기간, 문의처는
정책자료에서 확인되는 사실만 사용하세요.

9. 사업 전체 예산을
한 업체가 받을 수 있는 금액으로 쓰지 마세요.

10. 신청기간 정보는 원문 내용을 최대한 그대로
application_period에 적으세요.

11. 명확하게 접수기간이 종료된 사업이나
현재 날짜보다 과거의 접수 예정 시기만
기재되어 있는 사업은 선정하지 마세요.

12. 다만 다음과 같은 표현은
현재 종료됐다고 임의로 판단하지 마세요.

- 상시 접수
- 예산 소진 시까지
- 자금 소진 시까지
- 한도 소진 시까지
- 별도 통지 시까지

이 경우 현재 잔여한도를 알 수 없으므로
추천은 가능하되 신청상태는 확인이 필요합니다.

13. 신청기간이 확인되지 않으면
"신청기간 확인 필요"라고 쓰세요.

14. 전화번호는 근거자료에 실제 있는 경우에만 쓰세요.

15. 아래 네 조건을 판정하세요.

- 지역조건
- 사업자조건
- 지원목적
- 세부요건

status는 아래 세 값 중 하나만 사용하세요.

적합
확인 필요
부적합

16. 사용자가 알려주지 않았거나
정책자료만으로 확인할 수 없는 조건은
반드시 "확인 필요"입니다.

17. 명확한 부적합이 하나라도 있으면
그 사업은 추천하지 마세요.

18. 이유 설명은 eligibility에 작성하지 마세요.
label과 status만 작성하세요.

19. 금액은 정책자료의 금액을 그대로 사용해도 됩니다.
화면에서 사용자가 읽기 쉬운 단위로 별도 변환합니다.

20. 강원신용보증재단 운영상품도
다른 정책자료와 동일한 기준으로 비교하세요.
강원신용보증재단 상품이라는 이유만으로
추천하거나 제외하지 마세요.

반드시 JSON만 반환하세요.

형식:

{{
    "results": [
        {{
            "amount": "최대 5천만 원",
            "name": "공식 사업명",
            "type": "보증",
            "target": "지원대상 핵심내용",
            "contact": "기관명과 실제 전화번호",
            "source": "PDF 파일명",
            "page": "120",
            "application_period": "정책자료의 신청기간 정보",
            "eligibility": [
                {{
                    "label": "지역조건",
                    "status": "적합"
                }},
                {{
                    "label": "사업자조건",
                    "status": "확인 필요"
                }},
                {{
                    "label": "지원목적",
                    "status": "적합"
                }},
                {{
                    "label": "세부요건",
                    "status": "확인 필요"
                }}
            ]
        }}
    ]
}}

type은 아래 중 하나만 사용하세요.

지원금·보조금
인건비·고용지원
융자·대출
보증
R&D
컨설팅·서비스
기타

근거자료:

{context}
"""

    response = (
        gemini_client
        .models
        .generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
    )

    if not response.text:
        return []

    data = json.loads(
        clean_json_text(
            response.text
        )
    )

    results = []
    seen_names = set()

    for result in data.get(
        "results",
        []
    ):

        if not isinstance(
            result,
            dict
        ):
            continue

        name_key = normalize_text(
            result.get(
                "name",
                ""
            )
        )

        if (
            not name_key
            or name_key in seen_names
        ):
            continue

        validated = validate_result(
            result,
            candidates
        )

        if not validated:
            continue

        seen_names.add(
            name_key
        )

        results.append(
            validated
        )

    return results[:MAX_RESULTS]


# =========================================================
# 24. 추가 질문
# =========================================================

def needs_clarification(question):

    q = normalize_text(
        question
    )

    clear_terms = [
        "운영자금",
        "운전자금",
        "대출",
        "융자",
        "보증",
        "인건비",
        "직원",
        "채용",
        "마케팅",
        "홍보",
        "판로",
        "시설",
        "장비",
        "설비",
        "리모델링",
        "기술개발",
        "연구개발",
        "r&d",
        "수출",
        "창업",
        "예비창업",
    ]

    if any(
        term in q
        for term in clear_terms
    ):
        return False

    vague_terms = [
        "장사가 어렵",
        "장사가 안",
        "매출이 줄",
        "힘들",
        "어렵",
        "도움",
        "혜택이 있",
        "지원받",
        "무슨 지원",
        "뭐 받을",
    ]

    return any(
        term in q
        for term in vague_terms
    )


def make_clarified_question(
    original,
    choice
):

    additions = {

        "사업비·운영비":
            " 특히 사업 운영에 드는 비용을 줄일 수 있는 "
            "지원금이나 보조금이 필요합니다.",

        "대출·자금조달":
            " 특히 운영자금이나 정책자금 대출이 필요합니다.",

        "직원 인건비":
            " 특히 직원 채용이나 인건비 부담을 "
            "줄일 지원이 필요합니다.",

        "홍보·마케팅":
            " 특히 홍보, 마케팅, 판로 관련 지원이 필요합니다.",

        "시설·장비":
            " 특히 시설개선이나 장비·설비 관련 "
            "지원이 필요합니다.",

        "보증":
            " 특히 담보 부담을 줄이기 위한 "
            "신용보증 지원이 필요합니다.",

        "잘 모르겠어요":
            " 어떤 종류가 가장 적합한지 잘 모르겠으니 "
            "현재 상황에서 도움이 되는 지원을 찾아주세요.",
    }

    return (
        original.strip()
        + additions.get(
            choice,
            ""
        )
    )


def clarify_support_type(choice):

    mapping = {

        "사업비·운영비":
            "지원금",

        "대출·자금조달":
            "융자",

        "직원 인건비":
            "지원금",

        "홍보·마케팅":
            "지원금",

        "시설·장비":
            "전체",

        "보증":
            "보증",

        "잘 모르겠어요":
            "전체",
    }

    return mapping.get(
        choice,
        "전체"
    )


# =========================================================
# 25. 전화 질문 생성
# =========================================================

def generate_call_script(
    result,
    user_question
):

    name = result.get(
        "name",
        "해당 지원사업"
    )

    prompt = f"""
디지털 정보와 정책용어에 익숙하지 않은
소상공인이 담당기관에 전화할 때
화면을 보고 그대로 읽을 수 있는
짧은 전화 질문을 작성하세요.

사용자 상황:
{user_question}

지원사업:
{name}

규칙:

1. 정확히 4문장만 작성하세요.

2. 어려운 표현을 쓰지 마세요.

3. 첫 문장은 반드시 다음처럼 시작하세요.

"안녕하세요. {name} 보고 연락드렸습니다."

4. 두 번째 문장은 사용자의 상황을 반영해
지원대상인지 질문하세요.

5. 세 번째 문장은 지금도 신청 가능한지 질문하세요.

6. 네 번째 문장은 무엇을 준비해야 하는지 질문하세요.

7. 새로운 사실이나 조건을 만들지 마세요.

8. 번호와 글머리표를 사용하지 마세요.
"""

    try:

        response = (
            gemini_client
            .models
            .generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
        )

        if response.text:

            lines = [
                line.strip()
                for line in response.text.splitlines()
                if line.strip()
            ]

            if len(lines) >= 4:
                return lines[:4]

    except Exception:
        pass

    return [
        f"안녕하세요. {name} 보고 연락드렸습니다.",
        "제가 이 지원을 받을 수 있는 대상인지 궁금합니다.",
        "지금도 신청할 수 있나요?",
        "신청하려면 무엇을 준비하면 될까요?",
    ]


# =========================================================
# 26. 상태 디자인
# =========================================================

def condition_css(status):

    if status == "적합":
        return "status-good"

    if status == "부적합":
        return "status-bad"

    return "status-warn"


def application_css(status):

    if status in [
        "접수 중",
        "상시 접수",
    ]:
        return "status-good"

    if status in [
        "접수 종료",
        "기간 경과",
    ]:
        return "status-bad"

    return "status-warn"


# =========================================================
# 27. 처음으로
# =========================================================

def go_home():

    st.session_state.screen = "search"
    st.session_state.results = []
    st.session_state.result_index = 0
    st.session_state.original_question = ""
    st.session_state.clarify_question = ""
    st.session_state.selected_support = "전체"
    st.session_state.call_scripts = {}

    st.rerun()


# =========================================================
# 28. 검색
# =========================================================

def perform_search(
    question,
    support_type
):

    with st.spinner(
        "내 상황과 맞는 혜택을 확인하고 있습니다..."
    ):

        return generate_results(
            question,
            support_type
        )


# =========================================================
# 29. 첫 화면
# =========================================================

if st.session_state.screen == "search":

    st.title(
        "강원 소상공인 혜택 도우미"
    )

    render_html(
        """
<div class="hero-card">
<div class="hero-label">
강원지역 소상공인을 위한 AI 안내
</div>
<div class="hero-title">
어려운 정책 이름을 몰라도 괜찮습니다.
</div>
<div class="hero-text">
지금 상황을 평소 말하듯 적어주세요.<br>
받을 가능성이 있는 혜택을 찾아
담당기관까지 연결해드립니다.
</div>
</div>
"""
    )

    st.markdown(
        "### 어떤 도움이 필요하신가요?"
    )

    question = st.text_area(
        "사업 상황",
        placeholder=(
            "예: 춘천에서 식당을 운영하고 있는데 "
            "요즘 매출이 줄어서 도움이 필요해요."
        ),
        height=155,
        label_visibility="collapsed",
    )

    st.caption(
        "지역과 업종, 지금 어려운 점을 함께 적어주시면 "
        "더 정확하게 찾을 수 있습니다."
    )

    if st.button(
        "내가 받을 수 있는 혜택 찾기",
        use_container_width=True,
        type="primary",
    ):

        if not question.strip():

            st.warning(
                "현재 상황을 한 문장이라도 적어주세요."
            )

        else:

            region = detect_region(
                question
            )

            if (
                region
                and region != "강원"
            ):

                st.info(
                    "현재 서비스는 강원지역 소상공인을 "
                    "우선 대상으로 안내하고 있습니다."
                )

            elif needs_clarification(
                question
            ):

                st.session_state.clarify_question = (
                    question
                )

                st.session_state.screen = (
                    "clarify"
                )

                st.rerun()

            else:

                support_type = detect_support_type(
                    question
                )

                try:

                    results = perform_search(
                        question,
                        support_type
                    )

                    if not results:

                        st.warning(
                            "현재 확인 가능한 자료에서는 "
                            "신청 가능한 지원사업을 찾지 못했습니다."
                        )

                    else:

                        st.session_state.results = results
                        st.session_state.result_index = 0

                        st.session_state.original_question = (
                            question
                        )

                        st.session_state.selected_support = (
                            support_type
                        )

                        st.session_state.call_scripts = {}

                        st.session_state.screen = (
                            "results"
                        )

                        st.rerun()

                except Exception as e:

                    st.error(
                        "검색 중 오류가 발생했습니다."
                    )

                    st.code(
                        str(e)
                    )


# =========================================================
# 30. 추가 질문 화면
# =========================================================

elif st.session_state.screen == "clarify":

    st.title(
        "조금만 더 알려주세요"
    )

    render_html(
        """
<div class="hero-card">
<div class="hero-label">
한 번만 더 선택해주세요
</div>
<div class="hero-title">
지금 가장 필요한 도움은 무엇인가요?
</div>
<div class="hero-text">
정확한 정책 이름은 몰라도 괜찮습니다.<br>
가장 가까운 항목 하나만 눌러주세요.
</div>
</div>
"""
    )

    with st.expander(
        "내가 입력한 내용"
    ):

        st.write(
            st.session_state.clarify_question
        )

    col1, col2 = st.columns(2)

    with col1:

        b1 = st.button(
            "💰 사업비·운영비",
            use_container_width=True,
        )

    with col2:

        b2 = st.button(
            "🏦 대출·자금조달",
            use_container_width=True,
        )

    col3, col4 = st.columns(2)

    with col3:

        b3 = st.button(
            "👥 직원 인건비",
            use_container_width=True,
        )

    with col4:

        b4 = st.button(
            "📢 홍보·마케팅",
            use_container_width=True,
        )

    col5, col6 = st.columns(2)

    with col5:

        b5 = st.button(
            "🏪 시설·장비",
            use_container_width=True,
        )

    with col6:

        b6 = st.button(
            "🛡 보증",
            use_container_width=True,
        )

    b7 = st.button(
        "잘 모르겠어요",
        use_container_width=True,
    )

    choice = None

    if b1:
        choice = "사업비·운영비"

    elif b2:
        choice = "대출·자금조달"

    elif b3:
        choice = "직원 인건비"

    elif b4:
        choice = "홍보·마케팅"

    elif b5:
        choice = "시설·장비"

    elif b6:
        choice = "보증"

    elif b7:
        choice = "잘 모르겠어요"

    if choice:

        clarified_question = (
            make_clarified_question(
                st.session_state.clarify_question,
                choice,
            )
        )

        support_type = clarify_support_type(
            choice
        )

        try:

            results = perform_search(
                clarified_question,
                support_type,
            )

            if not results:

                st.warning(
                    "현재 확인 가능한 자료에서는 "
                    "신청 가능한 지원사업을 찾지 못했습니다."
                )

            else:

                st.session_state.results = results
                st.session_state.result_index = 0

                st.session_state.original_question = (
                    clarified_question
                )

                st.session_state.selected_support = (
                    support_type
                )

                st.session_state.call_scripts = {}

                st.session_state.screen = (
                    "results"
                )

                st.rerun()

        except Exception as e:

            st.error(
                "검색 중 오류가 발생했습니다."
            )

            st.code(
                str(e)
            )

    st.divider()

    if st.button(
        "← 질문 다시 쓰기",
        use_container_width=True,
    ):

        st.session_state.screen = (
            "search"
        )

        st.rerun()


# =========================================================
# 31. 결과 화면
# =========================================================

else:

    results = st.session_state.results

    if not results:
        go_home()

    index = st.session_state.result_index
    total = len(results)

    current = results[
        index
    ]

    st.caption(
        f"추천 결과 {index + 1} / {total}"
    )

    st.progress(
        (index + 1) / total
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

    raw_amount = current.get(
        "amount",
        "지원금액 확인 필요"
    )

    display_amount = (
        normalize_amount_display(
            raw_amount
        )
    )

    amount = html_escape(
        display_amount
    )

    name = html_escape(
        current.get(
            "name",
            "지원사업"
        )
    )

    type_label = TYPE_LABELS.get(
        current.get(
            "type",
            "기타"
        ),
        "지원사업"
    )

    render_html(
        f"""
<div class="result-card">
<div class="amount-label">
받을 수 있는 혜택
</div>
<div class="amount-main">
{amount}
</div>
<div class="policy-name">
{name}
</div>
<div class="type-pill">
{html_escape(type_label)}
</div>
</div>
"""
    )

    # =====================================================
    # 내 조건 확인
    # =====================================================

    eligibility = current.get(
        "eligibility",
        []
    )

    if eligibility:

        rows = ""

        for item in eligibility:

            label = html_escape(
                item.get(
                    "label",
                    "조건"
                )
            )

            status = item.get(
                "status",
                "확인 필요"
            )

            rows += (
                '<div class="condition-row">'
                '<div class="condition-label">'
                f'{label}'
                '</div>'
                '<div class="condition-status '
                f'{condition_css(status)}">'
                f'{html_escape(status)}'
                '</div>'
                '</div>'
            )

        render_html(
            f"""
<div class="info-card">
<div class="info-title">
내 조건 확인
</div>
{rows}
</div>
"""
        )

    # =====================================================
    # 신청 상태
    # =====================================================

    application_status = current.get(
        "application_status",
        "확인 필요"
    )

    application_detail = current.get(
        "application_detail",
        "담당기관에 현재 접수 여부를 확인해주세요."
    )

    render_html(
        f"""
<div class="application-card">
<div class="info-title">
신청상태
</div>
<div class="application-status {application_css(application_status)}">
{html_escape(application_status)}
</div>
<div class="application-detail">
{html_escape(application_detail)}
</div>
</div>
"""
    )

    # =====================================================
    # 전화
    # =====================================================

    contact = normalize_contact_display(
        current.get(
            "contact",
            "세부 공고 확인"
        )
    )

    render_html(
        f"""
<div class="phone-card">
<div class="phone-label">
전화로 확인하세요
</div>
<div class="phone-main">
{html_escape(contact)}
</div>
</div>
"""
    )

    phone = extract_phone(
        contact
    )

    if phone:

        tel_phone = re.sub(
            r"[^0-9+]",
            "",
            phone
        )

        render_html(
            f"""
<a
class="tel-button"
href="tel:{tel_phone}">
☎ 전화로 문의하기
</a>
"""
        )

    # =====================================================
    # 뭐라고 물어볼까요?
    # =====================================================

    script_key = (
        f"{index}:"
        f"{current.get('name', '')}:"
        f"{current.get('source', '')}"
    )

    if st.button(
        "💬 뭐라고 물어볼까요?",
        use_container_width=True,
        type="primary",
        key=f"call_help_{index}",
    ):

        with st.spinner(
            "전화할 때 읽을 말을 준비하고 있습니다..."
        ):

            script = generate_call_script(
                current,
                st.session_state.original_question,
            )

        st.session_state.call_scripts[
            script_key
        ] = script

    script = (
        st.session_state.call_scripts.get(
            script_key
        )
    )

    if script:

        script_html = ""

        for line in script:

            script_html += (
                '<div class="call-line">'
                f'“{html_escape(line)}”'
                '</div>'
            )

        render_html(
            f"""
<div class="call-card">
<div class="call-title">
📞 이렇게 말씀해보세요
</div>
{script_html}
</div>
"""
        )

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

        page = current.get(
            "page",
            "확인 필요"
        )

        st.write(
            f"{page}페이지"
        )

        st.caption(
            "실제 신청 전에는 담당기관의 최신 공고를 "
            "한 번 더 확인해주세요."
        )

    st.divider()

    # =====================================================
    # 이동
    # =====================================================

    col1, col2, col3 = st.columns(
        [
            1,
            1.25,
            1,
        ]
    )

    with col1:

        if st.button(
            "← 이전",
            use_container_width=True,
            disabled=(
                index == 0
            ),
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
        "※ 이 서비스는 지원사업 탐색을 돕는 안내 도구입니다. "
        "최종 신청 가능 여부는 담당기관에서 확인해주세요."
    )