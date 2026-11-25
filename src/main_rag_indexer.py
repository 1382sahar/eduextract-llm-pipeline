import os
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from d1_client import D1PipelineClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class EduExtractLayer4RAGIndexer:
    COLLECTION_NAME = "eduextract_knowledge_base"
    EMBEDDING_DIM = 768

    def __init__(self, qdrant_path: str = "artifacts/qdrant_db"):
        self.d1 = D1PipelineClient()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable!")

        self.client = genai.Client(api_key=api_key)
        self.embedding_model = self._select_embedding_model()
        logger.info(f"Layer 4 RAG Indexer initialized with embedding model: {self.embedding_model}")

        os.makedirs(qdrant_path, exist_ok=True)
        self.qdrant = QdrantClient(path=qdrant_path)
        self._ensure_qdrant_collection()

    def _select_embedding_model(self) -> str:
        try:
            available = [
                m.name.replace("models/", "")
                for m in self.client.models.list()
                if "embedding" in m.name.lower()
            ]
            for preferred in ("text-embedding-004", "gemini-embedding-001"):
                if preferred in available:
                    return preferred
            if available:
                return available[0]
        except Exception as e:
            logger.warning(f"Could not list embedding models dynamically: {e}")
        return "text-embedding-004"

    def _ensure_qdrant_collection(self):
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.COLLECTION_NAME not in existing:
            self.qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=self.EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection '{self.COLLECTION_NAME}' (dim={self.EMBEDDING_DIM}).")

    def _embed_batch(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """Generates 768-D embeddings in batches to respect Gemini API rate limits."""
        all_vectors: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=self.EMBEDDING_DIM),
                )
            except Exception:
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch,
                )

            for emb in response.embeddings:
                all_vectors.append([float(v) for v in emb.values[: self.EMBEDDING_DIM]])

            if i + batch_size < len(texts):
                time.sleep(1)

        return all_vectors

    @staticmethod
    def _deterministic_id(key: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def _build_chunks_for_video(self, vid_id: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        video_title = data.get("title") or vid_id
        topic = data.get("topic") or "General"
        sections = data.get("study_guide_sections") or []
        flashcards = data.get("flashcards") or []

        chunks: List[Dict[str, Any]] = []

        for idx, sec in enumerate(sections):
            sec_title = sec.get("section_title", f"Section {idx + 1}")
            summary = sec.get("summary", "").strip()
            takeaways = sec.get("key_takeaways", [])
            formulas = sec.get("formulas_or_definitions", [])

            parts = [
                f"Topic: {topic} | Lecture: {video_title} | Section: {sec_title}",
                f"Summary: {summary}",
            ]
            if takeaways:
                parts.append("Key Takeaways:\n- " + "\n- ".join(takeaways))
            if formulas:
                parts.append("Formulas & Definitions:\n* " + "\n* ".join(formulas))

            content = "\n".join(parts)
            chunks.append({
                "chunk_id": f"{vid_id}_sec_{idx}",
                "qdrant_id": self._deterministic_id(f"{vid_id}_sec_{idx}"),
                "video_id": vid_id,
                "video_title": video_title,
                "topic": topic,
                "section_title": sec_title,
                "chunk_type": "study_guide_section",
                "content": content,
            })

        if flashcards:
            cards_str = "\n".join(
                f"Q ({c.get('difficulty', 'Medium')}): {c.get('front', '')} | A: {c.get('back', '')}"
                for c in flashcards
            )
            fc_content = f"Topic: {topic} | Lecture: {video_title} | Review Flashcards:\n{cards_str}"
            chunks.append({
                "chunk_id": f"{vid_id}_flashcards",
                "qdrant_id": self._deterministic_id(f"{vid_id}_flashcards"),
                "video_id": vid_id,
                "video_title": video_title,
                "topic": topic,
                "section_title": "Review Flashcards",
                "chunk_type": "flashcards",
                "content": fc_content,
            })

        return chunks

    def run(self):
        videos = self.d1.get_videos(status="COMPLETED", rag_indexed=False)
        logger.info(f"Found {len(videos)} COMPLETED unindexed videos in D1.")

        indexed_videos = 0
        total_vectors = 0

        for data in videos:
            vid_id = data["id"]
            chunks = self._build_chunks_for_video(vid_id, data)
            if not chunks:
                logger.warning(f"[SKIP] {vid_id} has no study sections or flashcards to index.")
                continue

            title = data.get("title") or vid_id
            logger.info(f"Embedding {len(chunks)} chunks for '{title}' ({vid_id})...")

            try:
                vectors = self._embed_batch([c["content"] for c in chunks])
                now_iso = datetime.now(timezone.utc).isoformat()

                qdrant_points: List[qmodels.PointStruct] = []
                d1_vector_rows: List[Dict[str, Any]] = []

                for item, vec in zip(chunks, vectors):
                    qdrant_points.append(
                        qmodels.PointStruct(
                            id=item["qdrant_id"],
                            vector=vec,
                            payload={
                                "video_id": item["video_id"],
                                "video_title": item["video_title"],
                                "topic": item["topic"],
                                "section_title": item["section_title"],
                                "chunk_type": item["chunk_type"],
                                "content": item["content"],
                            },
                        )
                    )

                    d1_vector_rows.append({
                        "chunk_id": item["chunk_id"],
                        "video_id": item["video_id"],
                        "video_title": item["video_title"],
                        "topic": item["topic"],
                        "section_title": item["section_title"],
                        "chunk_type": item["chunk_type"],
                        "content": item["content"],
                        "embedding": vec,
                        "embedding_model": self.embedding_model,
                        "indexed_at": now_iso,
                    })

                self.qdrant.upsert(collection_name=self.COLLECTION_NAME, points=qdrant_points)
                self.d1.upsert_vectors(d1_vector_rows, batch_size=10)

                self.d1.update_video(vid_id, {
                    "rag_indexed": True,
                    "error_log": None,
                    "updated_at": now_iso,
                })

                indexed_videos += 1
                total_vectors += len(chunks)
                logger.info(f"[INDEXED] {vid_id} -> {len(chunks)} vectors stored in Qdrant & Cloudflare D1.")
                time.sleep(1.5)

            except Exception as e:
                logger.error(f"[FAILED] RAG indexing error for {vid_id}: {e}")
                self.d1.update_video(vid_id, {
                    "rag_indexed": False,
                    "error_log": f"RAG error: {str(e)[:950]}",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })

        logger.info(f"Layer 4 complete. Indexed {indexed_videos} videos ({total_vectors} total vectors).")


if __name__ == "__main__":
    indexer = EduExtractLayer4RAGIndexer()
    indexer.run()