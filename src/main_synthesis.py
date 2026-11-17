import os
import re
import csv
import json
import time
import base64
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
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


class StudyGuidePDF(FPDF):
    def __init__(self, video_title: str, topic: str):
        super().__init__()
        self.video_title = video_title
        self.topic = topic
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
        if self.unicode_enabled:
            return text
        return text.encode("latin-1", "replace").decode("latin-1")

    def header(self):
        self.set_font(self.font_family_name, "B", 10)
        self.set_text_color(100, 100, 100)
        header_text = self.safe_text(f"EduExtract Study Guide | Topic: {self.topic}")
        self.cell(0, 8, header_text, new_x="LMARGIN", new_y="NEXT", align="R")
        self.line(10, 18, 200, 18)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_family_name, "", 9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


class EduExtractLayer3Synthesizer:
    def __init__(self):
        self.db = self._init_firestore()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable!")

        self.client = genai.Client(api_key=api_key)
        self.model = self._select_active_gemini_model()
        logger.info(f"Initialized Gemini Layer 3 Synthesizer with model: {self.model}")

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
        """Selects gemini-3.8-flash or dynamically discovers the latest active Flash model."""
        try:
            available = [
                m.name.replace("models/", "")
                for m in self.client.models.list()
                if "flash" in m.name.lower()
                and "image" not in m.name.lower()
                and "audio" not in m.name.lower()
                and "tts" not in m.name.lower()
            ]
            logger.info(f"Available Gemini Flash models: {available}")
            if "gemini-3.8-flash" in available:
                return "gemini-3.8-flash"
            if available:
                return sorted(available, reverse=True)[0]
        except Exception as e:
            logger.warning(f"Could not list Gemini models dynamically: {e}")
        return "gemini-3.8-flash"

    def synthesize_chunk(self, chunk_text: str, video_title: str, topic: str, chunk_idx: int) -> ChunkSynthesis:
        prompt = (
            f"You are an expert university professor creating a structured study guide and flashcards "
            f"for a lecture titled '{video_title}' on the topic '{topic}'.\n"
            f"Analyze Transcript Chunk #{chunk_idx + 1} below and extract a clear academic summary, "
            "key exam takeaways, formal definitions/formulas, and deep conceptual flashcards.\n\n"
            f"Transcript:\n{chunk_text}"
        )

        for attempt in range(1, 4):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                        response_schema=ChunkSynthesis,
                    ),
                )
                return ChunkSynthesis.model_validate_json(response.text)
            except Exception as e:
                if attempt == 3:
                    raise
                logger.warning(f"Gemini retry {attempt}/3 on chunk {chunk_idx + 1}: {e}")
                time.sleep(10 * attempt)

        raise RuntimeError(f"Failed to synthesize chunk {chunk_idx + 1}")

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

        pdf.set_font(f_name, "B", 16)
        pdf.set_text_color(20, 40, 80)
        pdf.multi_cell(0, 10, pdf.safe_text(title))
        pdf.ln(4)

        for idx, sec in enumerate(sections, 1):
            pdf.set_font(f_name, "B", 13)
            pdf.set_text_color(30, 80, 140)
            pdf.multi_cell(0, 8, pdf.safe_text(f"{idx}. {sec.get('section_title', f'Section {idx}')}"))
            pdf.ln(2)

            pdf.set_font(f_name, "", 11)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 6, pdf.safe_text(sec.get("summary", "")))
            pdf.ln(3)

            takeaways = sec.get("key_takeaways", [])
            if takeaways:
                pdf.set_font(f_name, "B", 11)
                pdf.cell(0, 7, pdf.safe_text("Key Exam Takeaways:"), new_x="LMARGIN", new_y="NEXT")
                pdf.set_font(f_name, "", 10)
                for item in takeaways:
                    pdf.multi_cell(0, 6, pdf.safe_text(f"  • {item}"))
                pdf.ln(2)

            formulas = sec.get("formulas_or_definitions", [])
            if formulas:
                pdf.set_font(f_name, "B", 11)
                pdf.cell(0, 7, pdf.safe_text("Formulas & Definitions:"), new_x="LMARGIN", new_y="NEXT")
                pdf.set_font(f_name, "", 10)
                for form in formulas:
                    pdf.multi_cell(0, 6, pdf.safe_text(f"  ▪ {form}"))
                pdf.ln(2)

            pdf.ln(3)

        if flashcards:
            pdf.add_page()
            pdf.set_font(f_name, "B", 14)
            pdf.set_text_color(20, 40, 80)
            pdf.cell(0, 10, pdf.safe_text(f"Review Flashcards ({len(flashcards)} Cards)"), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

            for i, card in enumerate(flashcards, 1):
                pdf.set_font(f_name, "B", 10)
                pdf.set_text_color(40, 40, 40)
                q_text = f"Q{i} [{card.get('difficulty', 'Medium')}]: {card.get('front', '')}"
                pdf.multi_cell(0, 6, pdf.safe_text(q_text))

                pdf.set_font(f_name, "", 10)
                pdf.set_text_color(60, 60, 60)
                a_text = f"A: {card.get('back', '')}"
                pdf.multi_cell(0, 6, pdf.safe_text(a_text))
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
        docs = col_ref.where(filter=FieldFilter("status", "in", ["CLEANED", "SYNTHESIS_FAILED"])).stream()

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
            study_sections: List[Dict[str, Any]] = []
            all_flashcards: List[Dict[str, Any]] = []

            try:
                for idx, chunk_obj in enumerate(chunks):
                    chunk_text = chunk_obj.get("text", "")
                    if not chunk_text.strip():
                        continue

                    result = self.synthesize_chunk(chunk_text, video_title=title, topic=topic, chunk_idx=idx)
                    sec_dict = result.model_dump()
                    study_sections.append({
                        "chunk_index": idx,
                        "section_title": sec_dict["section_title"],
                        "summary": sec_dict["summary"],
                        "key_takeaways": sec_dict["key_takeaways"],
                        "formulas_or_definitions": sec_dict["formulas_or_definitions"]
                    })
                    all_flashcards.extend(sec_dict["flashcards"])
                    time.sleep(2)

                pdf_path = self.generate_pdf(vid_id, title, topic, study_sections, all_flashcards)
                csv_path = self.export_anki_csv(vid_id, all_flashcards)

                doc.reference.update({
                    "study_guide_sections": study_sections,
                    "flashcards": all_flashcards,
                    "flashcard_count": len(all_flashcards),
                    "llm_model_used": self.model,
                    "status": "COMPLETED",
                    "completed_at": datetime.now(timezone.utc).isoformat()
                })

                synthesized_count += 1
                logger.info(
                    f"[SUCCESS] {vid_id} -> Generated {len(study_sections)} study sections, "
                    f"{len(all_flashcards)} flashcards, and saved PDF to {pdf_path}"
                )

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
