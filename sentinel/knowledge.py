"""Local, SQLite FTS based retrieval for user supplied security references."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {".md", ".rst", ".txt", ".pdf"}
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 220
QUERY_STOPWORDS = {
    "assessment", "context", "json", "registered", "available", "operations",
    "tool", "tools", "model", "output", "return", "exactly", "schema",
    "only", "propose", "action", "actions", "sentinel", "summary",
}


class KnowledgeBase:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, source TEXT NOT NULL, part INTEGER NOT NULL, content TEXT NOT NULL)")
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(source, content, content='chunks', content_rowid='id', tokenize='unicode61')")
            db.execute("CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN INSERT INTO chunks_fts(rowid, source, content) VALUES (new.id, new.source, new.content); END")
            db.execute("CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN INSERT INTO chunks_fts(chunks_fts, rowid, source, content) VALUES ('delete', old.id, old.source, old.content); END")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _read(path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("PDF ingestion requires pypdf; install the project dependencies again.") from exc
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _chunks(text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + CHUNK_SIZE)
            if end < len(text):
                boundary = text.rfind(" ", start + CHUNK_SIZE // 2, end)
                if boundary > start:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(text):
                break
            start = max(start + 1, end - CHUNK_OVERLAP)
        return chunks

    def ingest(self, path: Path) -> dict[str, Any]:
        path = Path(path).expanduser().resolve()
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
        files = [p for p in files if p.suffix.lower() in SUPPORTED_SUFFIXES]
        added = skipped = 0
        errors: list[str] = []
        with self._connect() as db:
            for file in files:
                source = file.name if path.is_file() else file.relative_to(path).as_posix()
                try:
                    chunks = self._chunks(self._read(file))
                    if not chunks:
                        skipped += 1
                        continue
                    db.execute("DELETE FROM chunks WHERE source = ?", (source,))
                    db.executemany("INSERT INTO chunks(source, part, content) VALUES (?, ?, ?)", [(source, i + 1, chunk) for i, chunk in enumerate(chunks)])
                    added += len(chunks)
                except Exception as exc:
                    skipped += 1
                    errors.append(f"{file.name}: {exc}")
        return {"files": len(files), "chunks_added": added, "files_skipped": skipped, "errors": errors}

    def search(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        terms = [term for term in dict.fromkeys(re.findall(r"[A-Za-z0-9_]{3,}", query.lower())) if term not in QUERY_STOPWORDS][:12]
        if not terms:
            return []
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        with self._connect() as db:
            rows = db.execute(
                "SELECT c.source, c.part, c.content, bm25(chunks_fts) AS rank "
                "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (match, limit),
            ).fetchall()
        return [{"source": row["source"], "chunk": row["part"], "content": row["content"]} for row in rows]

    def count(self) -> int:
        with self._connect() as db:
            return int(db.execute("SELECT count(*) FROM chunks").fetchone()[0])
