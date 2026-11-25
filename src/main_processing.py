import os
import re
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

from transformers import pipeline
from d1_client import D1PipelineClient

try:
    from groq import Groq
except ImportError:
    Groq = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class SmartASRCorrector:
    """
    Self-learning ASR error corrector backed by a local JSON cache (cached across GitHub runs).
    Applies cached regex rules first (0 tokens), then optionally uses Groq JSON mode
    to discover new domain-specific phonetic typos and persists them to disk.
    """

    def __init__(self, cache_path: str = "artifacts/asr_glossary_cache.json"):
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.groq_api_key) if (Groq and self.groq_api_key) else None
        self.model = "llama-3.1-8b-instant"
        self._cache: Dict[str, Dict[str, str]] = self._load_all_cache()

    @staticmethod
    def _topic_key(topic: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_") or "general"

    def _load_all_cache(self) -> Dict[str, Dict[str, str]]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                logger.warning(f"Could not read ASR glossary cache: {e}")
        return {}

    def _save_all_cache(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not write ASR glossary cache: {e}")

    def load_cached_glossary(self, topic: str) -> Dict[str, str]:
        return dict(self._cache.get(self._topic_key(topic), {}))

    @staticmethod
    def apply_glossary(text: str, glossary: Dict[str, str]) -> str:
        for wrong_phrase in sorted(glossary.keys(), key=len, reverse=True):
            correct_term = glossary[wrong_phrase]
            pattern = re.compile(rf"\b{re.escape(wrong_phrase)}\b", re.IGNORECASE)
            text = pattern.sub(correct_term, text)
        return text

    def discover_new_errors(self, text: str, topic: str, existing_glossary: Dict[str, str]) -> Dict[str, str]:
        if not self.client:
            return {}

        sample_text = " ".join(text.split()[:3500])
        system_prompt = (
            f"You are an academic ASR (speech-to-text) error detector for the topic: '{topic}'.\n"
            "Scan the raw YouTube transcript and identify ONLY phonetic transcription mistakes "
            "of technical terms, formulas, or proper nouns (e.g., 'i can value' -> 'Eigenvalue', "
            "'real you' -> 'ReLU', '3 blue 1 brown' -> '3Blue1Brown').\n"
            "Do NOT include normal grammar or punctuation fixes. Output ONLY a valid JSON object "
            "with a single key 'corrections' mapping lowercase wrong phrases to exact correct terms.\n"
            "If no technical ASR errors exist, return {\"corrections\": {}}."
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sample_text},
                ],
            )
            data = json.loads(resp.choices[0].message.content)
            raw_new = data.get("corrections", {})
            return {
                k.lower().strip(): v.strip()
                for k, v in raw_new.items()
                if isinstance(k, str)
                and isinstance(v, str)
                and k.lower().strip() != v.lower().strip()
                and k.lower().strip() not in existing_glossary
            }
        except Exception as e:
            logger.warning(f"LLM ASR discovery skipped due to error: {e}")
            return {}

    def process(self, text: str, topic: str) -> str:
        glossary = self.load_cached_glossary(topic)
        if glossary:
            text = self.apply_glossary(text, glossary)

        new_mappings = self.discover_new_errors(text, topic, glossary)
        if new_mappings:
            logger.info(f"[CACHE UPDATE] Learned {len(new_mappings)} new ASR rules for '{topic}': {new_mappings}")
            text = self.apply_glossary(text, new_mappings)
            key = self._topic_key(topic)
            merged = self._cache.get(key, {})
            merged.update(new_mappings)
            self._cache[key] = merged
            self._save_all_cache()

        return text


class EduExtractLayer2Processor:
    """
    Layer 2 Pipeline (Cloudflare D1):
    1. Fetch INGESTED/READY transcripts from D1.
    2. Strip audio artifacts, spoken fillers, and stutters.
    3. Apply cached + self-learning ASR corrections.
    4. Restore punctuation & sentence capitalization locally via HuggingFace transformers pipeline.
    5. Split into sentence-aware overlapping semantic chunks and update D1 status to CLEANED.
    """

    def __init__(self):
        self.d1 = D1PipelineClient()
        self.asr_corrector = SmartASRCorrector()

        logger.info("Loading local punctuation model (oliverguhr/fullstop-punctuation-multilang-large)...")
        self.punct_pipe = pipeline(
            "token-classification",
            model="oliverguhr/fullstop-punctuation-multilang-large",
            aggregation_strategy="none",
        )

        fillers = [
            r"uh+", r"um+", r"er+", r"ah+",
            r"you know", r"i mean", r"sort of", r"kind of", r"so basically",
        ]
        self.filler_pattern = re.compile(r"\b(?:" + "|".join(fillers) + r")\b", re.IGNORECASE)

    def clean_noise_and_fillers(self, raw_text: str) -> str:
        text = re.sub(r"\[.*?\]|\(.*?\)", " ", raw_text)
        text = self.filler_pattern.sub("", text)
        text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    def _predict_punctuation_chunked(self, text: str, chunk_words: int = 250, overlap_words: int = 20) -> str:
        """
        Predicts punctuation labels in overlapping word windows (respecting the 512-token
        BERT/XLM-R context limit) and reconstructs punctuated text cleanly.
        """
        words = text.split()
        if not words:
            return ""

        result_words: List[str] = []
        step = max(1, chunk_words - overlap_words)

        for start in range(0, len(words), step):
            window_words = words[start : start + chunk_words]
            window_text = " ".join(window_words)
            predictions = self.punct_pipe(window_text)

            word_labels = ["0"] * len(window_words)
            char_spans = []
            cursor = 0
            for w in window_words:
                idx = window_text.find(w, cursor)
                char_spans.append((idx, idx + len(w)))
                cursor = idx + len(w)

            for pred in predictions:
                end_char = pred["end"]
                label = pred["entity"]
                for w_idx, (s_c, e_c) in enumerate(char_spans):
                    if s_c < end_char <= e_c:
                        word_labels[w_idx] = label
                        break

            emit_start = 0 if start == 0 else overlap_words
            for w_idx in range(emit_start, len(window_words)):
                w = window_words[w_idx]
                lbl = word_labels[w_idx]
                punct = lbl if lbl in {".", ",", "?", "-", ":"} else ""
                result_words.append(f"{w}{punct}")

            if start + chunk_words >= len(words):
                break

        return " ".join(result_words)

    def restore_punctuation_and_casing(self, text: str) -> str:
        punctuated = self._predict_punctuation_chunked(text)
        parts = re.split(r"([.!?]+\s+)", punctuated)
        restored = "".join(
            part[0].upper() + part[1:] if (i % 2 == 0 and part) else part
            for i, part in enumerate(parts)
        )
        return restored.strip()

    @staticmethod
    def semantic_sentence_chunker(
        text: str,
        target_words: int = 900,
        overlap_sentences: int = 2,
    ) -> List[Dict[str, Any]]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return []

        chunks: List[Dict[str, Any]] = []
        current_sentences: List[str] = []
        current_word_count = 0

        for sentence in sentences:
            s_words = len(sentence.split())

            if current_sentences and (current_word_count + s_words > target_words):
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    "chunk_index": len(chunks),
                    "text": chunk_text,
                    "word_count": len(chunk_text.split()),
                    "sentence_count": len(current_sentences),
                })
                current_sentences = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
                current_word_count = sum(len(s.split()) for s in current_sentences)

            current_sentences.append(sentence)
            current_word_count += s_words

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "chunk_index": len(chunks),
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "sentence_count": len(current_sentences),
            })

        return chunks

    def run(self):
        videos = self.d1.get_videos_by_statuses(["INGESTED", "READY"])
        logger.info(f"Layer 2 fetched {len(videos)} videos ready for cleaning from D1.")

        processed_count = 0
        for data in videos:
            vid_id = data["id"]
            raw_transcript = data.get("raw_transcript") or ""
            topic = data.get("topic") or "General"
            title = data.get("title") or vid_id

            if not raw_transcript or len(raw_transcript.strip()) < 50:
                logger.warning(f"[SKIP] {vid_id} has empty or invalid raw_transcript.")
                continue

            logger.info(f"Processing Layer 2 for: {title} ({vid_id})...")
            try:
                denoised_text = self.clean_noise_and_fillers(raw_transcript)
                asr_fixed_text = self.asr_corrector.process(denoised_text, topic=topic)
                cleaned_transcript = self.restore_punctuation_and_casing(asr_fixed_text)
                chunks = self.semantic_sentence_chunker(cleaned_transcript, target_words=900, overlap_sentences=2)

                # Clear raw_transcript once chunked so D1 rows remain compact (<1MB)
                self.d1.update_video(vid_id, {
                    "chunks": chunks,
                    "raw_transcript": None,
                    "char_count": len(cleaned_transcript),
                    "status": "CLEANED",
                    "error_log": None,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                processed_count += 1
                logger.info(
                    f"[SUCCESS] {vid_id} cleaned ({len(cleaned_transcript):,} chars) "
                    f"and split into {len(chunks)} semantic chunks in D1."
                )

            except Exception as e:
                logger.error(f"[FAILED] Layer 2 error for {vid_id}: {e}")
                self.d1.update_video(vid_id, {
                    "status": "CLEANING_FAILED",
                    "error_log": str(e)[:1000],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })

        logger.info(f"Layer 2 complete. Total videos processed: {processed_count}")


if __name__ == "__main__":
    processor = EduExtractLayer2Processor()
    processor.run()