from pathlib import Path
import hashlib

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================================================
# 기본 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data" / "raw"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "support_policy"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BATCH_SIZE = 100


# =========================================================
# 텍스트 나누기
# =========================================================

def split_text(text):

    text = " ".join(text.split())

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if len(chunk) >= 50:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# =========================================================
# 파일 고유 ID 생성
# =========================================================

def get_file_hash(file_path):

    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            data = f.read(1024 * 1024)

            if not data:
                break

            hasher.update(data)

    return hasher.hexdigest()


# =========================================================
# PDF 목록 확인
# =========================================================

pdf_files = sorted(PDF_DIR.glob("*.pdf"))

print()
print("=" * 60)
print("중소기업 정책자료 PDF 처리")
print("=" * 60)
print()

print(f"발견된 PDF: {len(pdf_files)}개")

if not pdf_files:

    print("PDF 파일이 없습니다.")
    raise SystemExit


# =========================================================
# ChromaDB 연결
# =========================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# =========================================================
# 이미 처리된 파일 확인
# =========================================================

existing = collection.get(
    include=["metadatas"]
)

processed_hashes = set()

for metadata in existing.get("metadatas", []):

    if metadata:

        file_hash = metadata.get("file_hash")

        if file_hash:
            processed_hashes.add(file_hash)


print(
    f"기존 DB 문서 조각: "
    f"{collection.count()}개"
)

print()


# =========================================================
# 임베딩 모델은 필요할 때만 로드
# =========================================================

embedding_model = None


def get_embedding_model():

    global embedding_model

    if embedding_model is None:

        print()
        print("AI 검색 모델을 불러오는 중...")

        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print("AI 검색 모델 준비 완료")

    return embedding_model


# =========================================================
# 통계
# =========================================================

new_pdf_count = 0
skip_count = 0
image_pdf_count = 0
failed_count = 0
new_chunk_count = 0

image_pdf_names = []


# =========================================================
# PDF 하나씩 처리
# =========================================================

for pdf_number, pdf_path in enumerate(
    pdf_files,
    start=1
):

    print()
    print("-" * 60)
    print(
        f"[{pdf_number}/{len(pdf_files)}] "
        f"{pdf_path.name}"
    )
    print("-" * 60)

    file_hash = get_file_hash(pdf_path)


    # -----------------------------------------------------
    # 이미 처리한 동일 파일이면 건너뜀
    # -----------------------------------------------------

    if file_hash in processed_hashes:

        print("이미 처리된 PDF → 건너뜀")

        skip_count += 1
        continue


    # -----------------------------------------------------
    # PDF 열기
    # -----------------------------------------------------

    try:

        reader = PdfReader(
            str(pdf_path)
        )

    except Exception as e:

        print(
            f"PDF 열기 실패: {e}"
        )

        failed_count += 1
        continue


    print(
        f"페이지 수: {len(reader.pages)}"
    )


    documents = []
    metadatas = []
    ids = []

    extracted_character_count = 0


    # -----------------------------------------------------
    # 페이지별 텍스트 추출
    # -----------------------------------------------------

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            text = page.extract_text() or ""

        except Exception:

            text = ""


        text = text.strip()

        extracted_character_count += len(text)


        if len(text) < 20:
            continue


        chunks = split_text(text)


        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            chunk_id = (
                f"{file_hash[:16]}_"
                f"{page_number}_"
                f"{chunk_number}"
            )

            documents.append(chunk)

            metadatas.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                    "file_hash": file_hash,
                }
            )

            ids.append(chunk_id)


    # -----------------------------------------------------
    # 이미지형 PDF 판별
    # -----------------------------------------------------

    if not documents:

        print(
            "⚠ 텍스트를 추출하지 못했습니다."
        )

        print(
            "→ 이미지형/스캔형 PDF 가능성이 높습니다."
        )

        image_pdf_count += 1

        image_pdf_names.append(
            pdf_path.name
        )

        continue


    print(
        f"텍스트 문자 수: "
        f"{extracted_character_count:,}"
    )

    print(
        f"생성된 문서 조각: "
        f"{len(documents)}개"
    )


    # -----------------------------------------------------
    # 임베딩
    # -----------------------------------------------------

    model = get_embedding_model()

    print(
        "검색용 데이터로 변환 중..."
    )

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()


    # -----------------------------------------------------
    # ChromaDB 저장
    # -----------------------------------------------------

    for start in range(
        0,
        len(documents),
        BATCH_SIZE
    ):

        end = start + BATCH_SIZE

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )


    print(
        f"저장 완료: "
        f"{len(documents)}개 조각"
    )


    processed_hashes.add(
        file_hash
    )

    new_pdf_count += 1

    new_chunk_count += len(
        documents
    )


# =========================================================
# 최종 결과
# =========================================================

print()
print()
print("=" * 60)
print("처리 결과")
print("=" * 60)

print(
    f"전체 PDF: {len(pdf_files)}개"
)

print(
    f"새로 처리한 PDF: {new_pdf_count}개"
)

print(
    f"이미 처리되어 건너뛴 PDF: {skip_count}개"
)

print(
    f"이미지형/텍스트 미추출 PDF: "
    f"{image_pdf_count}개"
)

print(
    f"처리 실패 PDF: {failed_count}개"
)

print(
    f"새로 추가된 문서 조각: "
    f"{new_chunk_count}개"
)

print(
    f"현재 ChromaDB 전체 문서 조각: "
    f"{collection.count()}개"
)


# =========================================================
# 이미지형 PDF 목록
# =========================================================

if image_pdf_names:

    print()
    print("=" * 60)
    print("OCR이 필요한 것으로 추정되는 PDF")
    print("=" * 60)

    for name in image_pdf_names:

        print(
            f"- {name}"
        )


print()
print("=" * 60)
print("작업 완료")
print("=" * 60)