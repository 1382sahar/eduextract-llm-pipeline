import os
import re
import csv
import json
import time
import base64
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fpdf import FPDF

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class Flashcard(BaseModel):
    front: str = Field(description="Clear conceptual question or mathematical prompt")
    back: str = Field(description="Concise, accurate answer or formula explanation")
    difficulty: str = Field(default="Medium", description="Easy, Medium, or Hard")


class ChunkSynthesis(BaseModel):
    section_title: str = Field(description="Descriptive academic title for this section")
    summary: str = Field(description="Clear, well-structured academic explanation (150-220 words)")
    key_takeaways: List[str] = Field(description="3 to 5 bullet-point exam takeaways")
    formulas_or_definitions: List[str] = Field(default_factory=list, description="Important formulas or definitions")
    flashcards: List[Flashcard] = Field(description="4 to 5 high-value flashcards")


class VideoSynthesis(BaseModel):
    sections: List[ChunkSynthesis] = Field(description="Ordered academic sections covering the entire lecture")


class LocalHuggingFaceBackup:
    """
    Lazy-loaded local LLM fallback (Qwen2.5-1.5B-Instruct) running purely on CPU
    without requiring `accelerate` or `device_map`.
    """
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(self):
        self.tokenizer = None
        self.model = None

    def _ensure_loaded(self):
        if self.model is not None:
            return
        logger.info(f"[LOCAL BACKUP] Loading cached/local model '{self.MODEL_ID}' on CPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            dtype=torch.float32
        )
        self.model.eval()
        logger.info("[LOCAL BACKUP] Local model ready.")

    @staticmethod
    def _extract_json_block(raw_text: str) -> dict:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in local model output.")
        return json.loads(match.group(0))

    def synthesize_chunk(self, chunk_text: str, video_title: str, topic: str, chunk_idx: int) -> ChunkSynthesis:
        self._ensure_loaded()

        trimmed_text = " ".join(chunk_text.split()[:550])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an academic professor. Output ONLY valid JSON matching this exact schema:\n"
                    '{"section_title": "str", "summary": "120-word explanation", '
                    '"key_takeaways": ["point 1", "point 2", "point 3"], '
                    '"formulas_or_definitions": ["formula 1"], '
                    '"flashcards": [{"front": "question", "back": "answer", "difficulty": "Medium"}]}'
                )
            },
            {
                "role": "user",
                "content": f"Lecture: {video_title} ({topic}) - Part {chunk_idx + 1}\n\nTranscript:\n{trimmed_text}"
            }
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=550,
                temperature=0.2,
                do_sample=True
            )

        output_ids = generated_ids[0][len(inputs.input_ids[0]):]
        raw_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        parsed = self._extract_json_block(raw_output)
        return ChunkSynthesis.model_validate(parsed)


class StudyGuidePDF(FPDF):
    def __init__(self, video_title: str, topic: str):
        super().__init__()
        self.video_title = video_title
        self.topic = topic
        self.set_margins(left=15, top=15, right=15)
        self.set_auto_page_break(auto=True, margin=15)
        self.unicode_enabled = self._setup_fonts()

    def _setup_fonts(self) -> bool:
        regular_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if os.path.exists(regular_path) and os.path.exists(bold_path):
            self.add_font("DejaVu", "", regular_path)
            self.add_font("DejaVu", "B", bold_path)
            self.font_family_name = "DejaVu"
            return True
        self.font_family_name = "Helvetica"
        return False

    def safe_text(self, text: str) -> str:
        # شکستن کلمات/فرمول‌های فوق‌طولانی بدون فاصله برای جلوگیری از سرریز عرض صفحه در fpdf2
        cleaned = re.sub(r"(\S{55})", r"\1 ", str(text))
        if self.unicode_enabled:
            return cleaned
        return cleaned.encode("latin-1", "replace").decode("latin-1")

    def write_block(self, text: str, line_height: float = 6.0):
        """Always resets X to left margin and uses explicit effective page width (`epw`)."""
        self.set_x(self.l_margin)
        self.multi_cell(self.epw, line_height, self.safe_text(text))

    def header(self):
        self.set_x(self.l_margin)
        self.set_font(self.font_family_name, "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(self.epw, 8, self.safe_text(f"EduExtract Study Guide | Topic: {self.topic}"), align="R")
        self.ln(9)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_x(self.l_margin)
        self.set_font(self.font_family_name, "", 9)
        self.set_text_color(128, 128, 128)
        self.cell(self.epw, 10, f"Page {self.page_no()}", align="C")


class EduExtractLayer3Synthesizer:
    def __init__(self):
        self.db = self._init_firestore()
        api_key = os.getenv("GEMINI_API_KEY")
        self.client: Optional[genai.Client] = genai.Client(api_key=api_key) if api_key else None
        self.model = self._select_active_gemini_model() if self.client else None
        self.local_backup = LocalHuggingFaceBackup()
        self.gemini_quota_exhausted = False

        if self.model:
            logger.info(f"Primary Synthesizer: Gemini ({self.model}) | Backup: {LocalHuggingFaceBackup.MODEL_ID}")
        else:
            logger.warning("GEMINI_API_KEY missing; running directly on Local Backup model.")

        os.makedirs("artifacts/pdf", exist_ok=True)
        os.makedirs("artifacts/flashcards", exist_ok=True)

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

    def _select_active_gemini_model(self) -> str:
        try:
            available = [
                m.name.replace("models/", "")
                for m in self.client.models.list()
                if "flash" in m.name.lower()
                and "image" not in m.name.lower()
                and "audio" not in m.name.lower()
                and "tts" not in m.name.lower()
            ]
            if "gemini-3.8-flash" in available:
                return "gemini-3.8-flash"
            if available:
                return sorted(available, reverse=True)[0]
        except Exception as e:
            logger.warning(f"Could not list Gemini models dynamically: {e}")
        return "gemini-3.8-flash"

    def _synthesize_whole_video_with_gemini(
        self, chunks: List[Dict[str, Any]], video_title: str, topic: str
    ) -> VideoSynthesis:
        """
        Sends all chunks of a video in ONE single Gemini request to conserve the 20 RPD free-tier quota.
        """
        if not self.client or not self.model or self.gemini_quota_exhausted:
            raise RuntimeError("Gemini unavailable or daily quota exhausted.")

        combined_transcript = "\n\n".join(
            f"--- PART {i + 1} ---\n{c.get('text', '')}"
            for i, c in enumerate(chunks)
            if c.get("text", "").strip()
        )

        prompt = (
            f"You are an expert university professor creating a complete structured study guide and flashcards "
            f"for a lecture titled '{video_title}' on the topic '{topic}'.\n"
            f"Below is the full lecture divided into {len(chunks)} parts. For EACH part, produce one structured section "
            "with a descriptive title, a clear 150-220 word academic summary, 3-5 key exam takeaways, "
            "important formulas/definitions, and 4-5 conceptual flashcards.\n\n"
            f"{combined_transcript}"
        )

        for attempt in range(1, 3):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                        response_schema=VideoSynthesis,
                    ),
                )
                return VideoSynthesis.model_validate_json(response.text)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    self.gemini_quota_exhausted = True
                    raise RuntimeError(f"Gemini daily quota exhausted: {e}")
                if attempt == 2:
                    raise
                logger.warning(f"Gemini transient error (attempt {attempt}/2): {e}")
                time.sleep(6)

        raise RuntimeError("Gemini whole-video synthesis failed.")

    def synthesize_video(
        self, chunks: List[Dict[str, Any]], video_title: str, topic: str
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """Tries 1-shot Gemini synthesis first; falls back to chunk-by-chunk local Qwen2.5-1.5B if needed."""
        try:
            video_res = self._synthesize_whole_video_with_gemini(chunks, video_title, topic)
            study_sections = []
            all_flashcards = []
            for idx, sec in enumerate(video_res.sections):
                sec_dict = sec.model_dump()
                study_sections.append({
                    "chunk_index": idx,
                    "section_title": sec_dict["section_title"],
                    "summary": sec_dict["summary"],
                    "key_takeaways": sec_dict["key_takeaways"],
                    "formulas_or_definitions": sec_dict["formulas_or_definitions"]
                })
                all_flashcards.extend(sec_dict["flashcards"])
            return study_sections, all_flashcards, self.model

        except Exception as gemini_err:
            logger.warning(
                f"[FALLBACK TRIGGERED] Switching to local backup ({LocalHuggingFaceBackup.MODEL_ID}) "
                f"because Gemini returned: {gemini_err}"
            )
            study_sections = []
            all_flashcards = []
            for idx, chunk_obj in enumerate(chunks):
                chunk_text = chunk_obj.get("text", "")
                if not chunk_text.strip():
                    continue
                sec = self.local_backup.synthesize_chunk(chunk_text, video_title, topic, idx)
                sec_dict = sec.model_dump()
                study_sections.append({
                    "chunk_index": idx,
                    "section_title": sec_dict["section_title"],
                    "summary": sec_dict["summary"],
                    "key_takeaways": sec_dict["key_takeaways"],
                    "formulas_or_definitions": sec_dict["formulas_or_definitions"]
                })
                all_flashcards.extend(sec_dict["flashcards"])
            return study_sections, all_flashcards, f"local:{LocalHuggingFaceBackup.MODEL_ID}"

    @staticmethod
    def generate_pdf(
        vid_id: str,
        title: str,
        topic: str,
        sections: List[Dict[str, Any]],
        flashcards: List[Dict[str, Any]]
    ) -> str:
        pdf = StudyGuidePDF(video_title=title, topic=topic)
        pdf.add_page()
        f_name = pdf.font_family_name

        pdf.set_font(f_name, "B", 15)
        pdf.set_text_color(20, 40, 80)
        pdf.write_block(title, line_height=9)
        pdf.ln(4)

        for idx, sec in enumerate(sections, 1):
            pdf.set_font(f_name, "B", 12)
            pdf.set_text_color(30, 80, 140)
            pdf.write_block(f"{idx}. {sec.get('section_title', f'Section {idx}')}", line_height=8)
            pdf.ln(2)

            pdf.set_font(f_name, "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.write_block(sec.get("summary", ""), line_height=6)
            pdf.ln(3)

            takeaways = sec.get("key_takeaways", [])
            if takeaways:
                pdf.set_font(f_name, "B", 10)
                pdf.write_block("Key Exam Takeaways:", line_height=6)
                pdf.set_font(f_name, "", 10)
                for item in takeaways:
                    pdf.write_block(f"  - {item}", line_height=6)
                pdf.ln(2)

            formulas = sec.get("formulas_or_definitions", [])
            if formulas:
                pdf.set_font(f_name, "B", 10)
                pdf.write_block("Formulas & Definitions:", line_height=6)
                pdf.set_font(f_name, "", 10)
                for form in formulas:
                    pdf.write_block(f"  * {form}", line_height=6)
                pdf.ln(2)

            pdf.ln(3)

        if flashcards:
            pdf.add_page()
            pdf.set_font(f_name, "B", 14)
            pdf.set_text_color(20, 40, 80)
            pdf.write_block(f"Review Flashcards ({len(flashcards)} Cards)", line_height=8)
            pdf.ln(3)

            for i, card in enumerate(flashcards, 1):
                pdf.set_font(f_name, "B", 10)
                pdf.set_text_color(40, 40, 40)
                pdf.write_block(f"Q{i} [{card.get('difficulty', 'Medium')}]: {card.get('front', '')}", line_height=6)

                pdf.set_font(f_name, "", 10)
                pdf.set_text_color(60, 60, 60)
                pdf.write_block(f"A: {card.get('back', '')}", line_height=6)
                pdf.ln(3)

        safe_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", title)[:45].strip("_")
        out_path = f"artifacts/pdf/{vid_id}_{safe_slug}.pdf"
        pdf.output(out_path)
        return out_path

    @staticmethod
    def export_anki_csv(vid_id: str, flashcards: List[Dict[str, Any]]) -> str:
        csv_path = f"artifacts/flashcards/{vid_id}_anki_cards.csv"
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for card in flashcards:
                writer.writerow([card.get("front", ""), card.get("back", ""), card.get("difficulty", "Medium")])
        return csv_path

    def run(self):
        col_ref = self.db.collection("Pipeline_State")
        docs = col_ref.where(filter=FieldFilter("status", "in", ["CLEANED", "SYNTHESIS_FAILED"])).get()

        synthesized_count = 0
        for doc in docs:
            data = doc.to_dict()
            vid_id = doc.id
            title = data.get("title", vid_id)
            topic = data.get("topic", "General")
            chunks = data.get("chunks", [])

            if not chunks:
                logger.warning(f"[SKIP] {vid_id} has no semantic chunks.")
                continue

            logger.info(f"Synthesizing Layer 3 for: {title} ({vid_id}) - {len(chunks)} chunks...")
            try:
                study_sections, all_flashcards, used_model = self.synthesize_video(
                    chunks, video_title=title, topic=topic
                )

                pdf_path = self.generate_pdf(vid_id, title, topic, study_sections, all_flashcards)
                csv_path = self.export_anki_csv(vid_id, all_flashcards)

                doc.reference.update({
                    "study_guide_sections": study_sections,
                    "flashcards": all_flashcards,
                    "flashcard_count": len(all_flashcards),
                    "llm_model_used": used_model,
                    "status": "COMPLETED",
                    "completed_at": datetime.now(timezone.utc).isoformat()
                })

                synthesized_count += 1
                logger.info(
                    f"[SUCCESS] {vid_id} ({used_model}) -> Generated {len(study_sections)} study sections, "
                    f"{len(all_flashcards)} flashcards, and saved PDF to {pdf_path}"
                )
                time.sleep(2)

            except Exception as e:
                logger.error(f"[FAILED] Layer 3 synthesis error for {vid_id}: {e}")
                doc.reference.update({
                    "status": "SYNTHESIS_FAILED",
                    "error_log": str(e),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                })

        logger.info(f"Layer 3 complete. Total videos synthesized: {synthesized_count}")


if __name__ == "__main__":
    synthesizer = EduExtractLayer3Synthesizer()
    synthesizer.run()
