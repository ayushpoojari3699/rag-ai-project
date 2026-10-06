"""Index every PDF in the docs/ folder from the command line.

Usage:  python ingest.py
"""

from rag_core import DOCS_DIR, RAGEngine


def main():
    engine = RAGEngine()
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}/")
        return
    for pdf in pdfs:
        try:
            info = engine.add_pdf(pdf)
            print(f"Indexed {info['file']}: {info['pages']} pages -> {info['chunks']} chunks")
        except ValueError as e:
            print(f"Skipped {pdf.name}: {e}")
    print(f"Done. {len(engine.documents())} document(s) in the index.")


if __name__ == "__main__":
    main()
