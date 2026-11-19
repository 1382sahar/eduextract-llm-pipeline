# src/main_rag_indexer.py
import os
import json
import time
import uuid
import base64
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class EduExtractLayer4RAGIndexer:
    COLLECTION_NAME = "eduextract_knowledge_base"
    EMBEDDING_DIM = 768

    def __init__(self, qdrant_path: str = "artifacts/qdrant_db"):
        self.db = self._init_firestore()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable!")

        self.client = genai.Client(api_key=api_key)
        self.embedding_model = self._select_embedding_model()
        logger.info(f"Layer 4 RAG Indexer initialized with embedding model: {self.embedding_model}")

        os.makedirs(qdrant_path, exist_ok=True)
        self.qdrant = QdrantClient(path=qdrant_path)
        self._ensure_qdrant_collection()

    @staticmethod
    def _init_firestore() -> firestore.Client:
        if firebase_admin._apps:
            return firestore.client()

        firebase_b64 = os.getenv("FIREBASE_CREDENTIALS_B64")
        if firebase_b64:
            cred_dict = json.loads(base64.b64decode(firebase_b64).decode("utf-8"))
            cred = credentials.Certificate(cred_dict)
        elif os.path.exists("serviceAccountKey.json"):
            cred = credentials.Certificate("serviceAccountKey.json")
        else:
            raise ValueError("Missing FIREBASE_CREDENTIALS_B64 secret or local serviceAccountKey.json!")

        firebase_admin.initialize_app(cred)
        return firestore.client()

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
        """تولید بردار ۷۶۸ بُعدی به صورت دسته‌ای برای جلوگیری از محدودیت نرخ درخواست"""
        all_vectors: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
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
                all_vectors.append([float(v) for v in emb.values[:self.EMBEDDING_DIM]])

            if i + batch_size < len(texts):
                time.sleep(1)

        return all_vectors

    @staticmethod
    def _deterministic_id(key: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def _build_chunks_for_video(self, vid_id: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        video_title = data.get("title", vid_id)
        topic = data.get("topic", "General")
        sections = data.get("study_guide_sections", [])
        flashcards = data.get("flashcards", [])

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
        state_col = self.db.collection("Pipeline_State")
        vector_col = self.db.collection("RAG_Vectors")

        docs = list(state_col.where(filter=FieldFilter("status", "==", "COMPLETED")).get())
        logger.info(f"Found {len(docs)} COMPLETED videos. Checking RAG index state...")

        indexed_videos = 0
        total_vectors = 0

        for doc in docs:
            data = doc.to_dict()
            vid_id = doc.id

            if data.get("rag_indexed") is True:
                logger.info(f"[SKIP] {vid_id} already indexed.")
                continue

            chunks = self._build_chunks_for_video(vid_id, data)
            if not chunks:
                logger.warning(f"[SKIP] {vid_id} has no study sections or flashcards to index.")
                continue

            title = data.get("title", vid_id)
            logger.info(f"Embedding {len(chunks)} chunks for '{title}' ({vid_id})...")

            try:
                vectors = self._embed_batch([c["content"] for c in chunks])
                qdrant_points: List[qmodels.PointStruct] = []
                batch = self.db.batch()
                now_iso = datetime.now(timezone.utc).isoformat()

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

                    vec_ref = vector_col.document(item["chunk_id"])
                    batch.set(vec_ref, {
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
                batch.commit()

                doc.reference.update({
                    "rag_indexed": True,
                    "rag_chunk_count": len(chunks),
                    "rag_indexed_at": now_iso,
                    "updated_at": now_iso,
                }, timeout=60.0)

                indexed_videos += 1
                total_vectors += len(chunks)
                logger.info(f"[INDEXED] {vid_id} -> {len(chunks)} vectors stored in Qdrant & Firestore.")
                time.sleep(1.5)

            except Exception as e:
                logger.error(f"[FAILED] RAG indexing error for {vid_id}: {e}")
                doc.reference.update({
                    "rag_indexed": False,
                    "rag_error_log": str(e),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }, timeout=60.0)

        logger.info(f"Layer 4 complete. Indexed {indexed_videos} videos ({total_vectors} total vectors).")


if __name__ == "__main__":
    indexer = EduExtractLayer4RAGIndexer()
    indexer.run()