import os
import re
import glob
import json
import base64
import logging
import tempfile
import subprocess
from http.cookiejar import MozillaCookieJar
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests
import yt_dlp
import firebase_admin
from firebase_admin import credentials, firestore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class EduExtractIngestionPipeline:
    """
    Unified Layer 0 (Discovery via YouTube API v3) & Layer 1 (Authenticated Transcript Ingestion)
    integrated with Firebase Firestore state management and dynamic web dashboard config.
    """

    SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

    def __init__(self, languages: List[str] = ["en", "fa"]):
        self.languages = languages
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        self.firebase_b64 = os.getenv("FIREBASE_CREDENTIALS_B64")
        self.cookies_b64 = os.getenv("YOUTUBE_COOKIES_B64")
        self.proxy_url = os.getenv("YOUTUBE_PROXY_URL")

        if not self.youtube_api_key:
            raise ValueError("Missing YOUTUBE_API_KEY environment variable!")

        self.db = self._init_firestore()
        self.cookie_file_path = self._prepare_cookie_file()

    def _init_firestore(self) -> firestore.Client:
        if firebase_admin._apps:
            return firestore.client()

        if self.firebase_b64:
            logger.info("Authenticating Firebase via FIREBASE_CREDENTIALS_B64 secret...")
            cred_dict = json.loads(base64.b64decode(self.firebase_b64).decode("utf-8"))
            cred = credentials.Certificate(cred_dict)
        elif os.path.exists("serviceAccountKey.json"):
            logger.info("Authenticating Firebase via local serviceAccountKey.json...")
            cred = credentials.Certificate("serviceAccountKey.json")
        else:
            raise ValueError("Missing FIREBASE_CREDENTIALS_B64 secret or local serviceAccountKey.json!")

        firebase_admin.initialize_app(cred)
        return firestore.client()

    def get_remote_config(self) -> tuple[str, int]:
        """
        Fetches the pipeline configuration requested by the user via the Web Dashboard.
        Falls back to environment variables if the remote config is missing or invalid.
        """
        try:
            config_ref = self.db.collection("System").document("PipelineConfig")
            snapshot = config_ref.get()
            
            if snapshot.exists:
                data = snapshot.to_dict()
                topic = data.get("target_topic")
                top_k = data.get("top_k_videos", 5)
                
                if topic:
                    # برای امنیت بیشتر، تعداد ویدیوها را به حداکثر 10 محدود می‌کنیم
                    top_k = min(max(int(top_k), 1), 10)
                    logger.info(f"[CONFIG] Loaded config from Web Dashboard -> Topic: '{topic}', Count: {top_k}")
                    return topic, top_k
                    
        except Exception as e:
            logger.error(f"[CONFIG] Error reading remote config: {e}. Falling back to env vars.")
            
        # Fallback to env vars
        fallback_topic = os.getenv("TARGET_TOPIC", "Linear Algebra")
        fallback_k = int(os.getenv("TOP_K_VIDEOS", "5"))
        logger.info(f"[CONFIG] Using fallback env config -> Topic: '{fallback_topic}', Count: {fallback_k}")
        return fallback_topic, fallback_k

    def _prepare_cookie_file(self) -> Optional[str]:
        if not self.cookies_b64 or not self.cookies_b64.strip():
            logger.warning("YOUTUBE_COOKIES_B64 not set. Proceeding without cookies.")
            return None

        try:
            decoded = base64.b64decode(self.cookies_b64).decode("utf-8")
            temp_cookie = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8")
            if not decoded.lstrip().startswith("# Netscape HTTP Cookie File"):
                temp_cookie.write("# Netscape HTTP Cookie File\n")
            temp_cookie.write(decoded)
            temp_cookie.close()
            logger.info("Loaded YouTube cookies from YOUTUBE_COOKIES_B64 secret.")
            return temp_cookie.name
        except Exception as e:
            logger.error(f"Cookie decode error: {e}")
            return None

    def _build_authenticated_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        })
        if self.cookie_file_path:
            cj = MozillaCookieJar(self.cookie_file_path)
            cj.load(ignore_discard=True, ignore_expires=True)
            session.cookies = cj
        if self.proxy_url:
            session.proxies = {"http": self.proxy_url, "https": self.proxy_url}
        return session

    @staticmethod
    def _parse_iso_duration(duration_str: str) -> float:
        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
        if not match:
            return 0.0
        h, m, s = int(match.group(1) or 0), int(match.group(2) or 0), int(match.group(3) or 0)
        return round(h * 60 + m + (s / 60), 1)

    def discover_top_videos(self, topic: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Searching YouTube API v3 for topic: '{topic}'...")
        search_params = {
            "part": "snippet",
            "q": f"{topic} lecture tutorial explained",
            "type": "video",
            "videoCaption": "closedCaption",
            "relevanceLanguage": "en",
            "order": "viewCount",
            "maxResults": 20,
            "key": self.youtube_api_key
        }
        resp = requests.get(self.SEARCH_URL, params=search_params, timeout=15)
        resp.raise_for_status()
        video_ids = [item["id"]["videoId"] for item in resp.json().get("items", []) if "videoId" in item.get("id", {})]

        if not video_ids:
            return []

        stats_params = {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(video_ids),
            "key": self.youtube_api_key
        }
        stats_resp = requests.get(self.VIDEOS_URL, params=stats_params, timeout=15)
        stats_resp.raise_for_status()

        candidates = []
        for item in stats_resp.json().get("items", []):
            vid_id = item["id"]
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            duration_min = self._parse_iso_duration(item.get("contentDetails", {}).get("duration", ""))

            if duration_min < 4.0:
                continue

            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            score = views + (likes * 25)

            candidates.append({
                "video_id": vid_id,
                "title": snippet.get("title"),
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "channel": snippet.get("channelTitle"),
                "view_count": views,
                "like_count": likes,
                "duration_minutes": duration_min,
                "engagement_score": score,
                "topic": topic
            })

        ranked = sorted(candidates, key=lambda x: x["engagement_score"], reverse=True)
        return ranked[:top_k]

    def _fetch_ytdlp_python(self, video_id: str) -> str:
        """
        Engine 1: yt-dlp Python API with ignore_no_formats_error=True so it never fails
        when video streams are restricted on datacenter IPs.
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts: Dict[str, Any] = {
            "skip_download": True,
            "ignore_no_formats_error": True,
            "check_formats": False,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en.*", "en", "fa"],
            "quiet": True,
            "no_warnings": True,
        }

        if self.cookie_file_path:
            ydl_opts["cookiefile"] = self.cookie_file_path

        if self.proxy_url:
            ydl_opts["proxy"] = self.proxy_url

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise ValueError(f"Empty info dict returned for {video_id}")

            subs = info.get("subtitles") or {}
            auto_subs = info.get("automatic_captions") or {}

            tracks = None
            # Check manual English subtitles first
            for key in subs:
                if key == "en" or key.startswith("en-"):
                    tracks = subs[key]
                    break
            # Then check automatic English captions
            if not tracks:
                for key in auto_subs:
                    if key == "en" or key.startswith("en-"):
                        tracks = auto_subs[key]
                        break
            # Fallback to any available subtitle track
            if not tracks:
                all_tracks = list(subs.values()) + list(auto_subs.values())
                if all_tracks:
                    tracks = all_tracks[0]

            if not tracks:
                raise ValueError(f"No subtitle tracks found in info dict for {video_id}")

            json3_url = next((t["url"] for t in tracks if t.get("ext") == "json3"), tracks[0]["url"])
            session = self._build_authenticated_session()
            resp = session.get(json3_url, timeout=15)
            resp.raise_for_status()

            if "json3" in json3_url or resp.text.lstrip().startswith("{"):
                sub_data = resp.json()
                words = [
                    seg.get("utf8", "").strip()
                    for event in sub_data.get("events", [])
                    for seg in event.get("segs", [])
                    if seg.get("utf8", "").strip() and seg.get("utf8") != "\n"
                ]
                text = " ".join(words)
            else:
                clean_text = re.sub(r"<[^>]+>", " ", resp.text)
                text = " ".join(clean_text.split())

            if len(text) < 50:
                raise ValueError("Extracted transcript is too short/empty.")
            return text

    def _fetch_ytdlp_cli_fallback(self, video_id: str) -> str:
        """
        Engine 2: Direct CLI execution of yt-dlp writing .vtt subtitle file to disk
        with --ignore-no-formats-error.
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        with tempfile.TemporaryDirectory() as tmpdir:
            out_template = os.path.join(tmpdir, "%(id)s.%(ext)s")
            cmd = [
                "yt-dlp",
                "--skip-download",
                "--ignore-no-formats-error",
                "--write-subs",
                "--write-auto-subs",
                "--sub-langs", "en.*,en",
                "--sub-format", "vtt",
                "-o", out_template,
                url
            ]
            if self.cookie_file_path:
                cmd.extend(["--cookies", self.cookie_file_path])

            subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            vtt_files = glob.glob(os.path.join(tmpdir, "*.vtt"))
            if not vtt_files:
                raise RuntimeError(f"yt-dlp CLI did not produce any .vtt file for {video_id}")

            with open(vtt_files[0], "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            lines = []
            for line in content.splitlines():
                line = line.strip()
                if (
                    not line
                    or line.startswith("WEBVTT")
                    or line.startswith("Kind:")
                    or line.startswith("Language:")
                    or "-->" in line
                    or line.isdigit()
                ):
                    continue
                clean_line = re.sub(r"<[^>]+>", "", line).strip()
                if clean_line and (not lines or clean_line != lines[-1]):
                    lines.append(clean_line)

            full_text = " ".join(lines)
            if len(full_text) < 50:
                raise ValueError("CLI VTT transcript was empty.")
            return full_text

    def extract_transcript(self, video_id: str) -> tuple[str, str]:
        """Tries Engine 1 (yt-dlp Python API) -> Engine 2 (yt-dlp CLI Disk Writer)."""
        try:
            return self._fetch_ytdlp_python(video_id), "yt-dlp-python"
        except Exception as e1:
            logger.warning(f"Engine 1 failed for {video_id} ({type(e1).__name__}: {e1}). Trying Engine 2 (yt-dlp CLI)...")
            return self._fetch_ytdlp_cli_fallback(video_id), "yt-dlp-cli"

    def run(self, topic: str, top_k: int = 5):
        top_videos = self.discover_top_videos(topic=topic, top_k=top_k)
        col_ref = self.db.collection("Pipeline_State")

        for vid in top_videos:
            vid_id = vid["video_id"]
            doc_ref = col_ref.document(vid_id)
            snapshot = doc_ref.get()

            # ==============================================================
            # SYSTEM STATE LOCK: جلوگیری از رونویسی ویدیوهای پردازش‌شده
            # ==============================================================
            if snapshot.exists:
                current_status = snapshot.to_dict().get("status", "UNKNOWN")
                if current_status in ["READY", "INGESTED", "EXTRACTED", "CLEANED", "SYNTHESIS_FAILED", "COMPLETED"]:
                    logger.info(f"[SKIP] Video {vid_id} already in state: '{current_status}'. Preserving lifecycle.")
                    continue
            # ==============================================================

            logger.info(f"Extracting transcript for: {vid['title']} ({vid_id})...")
            try:
                transcript_text, engine = self.extract_transcript(vid_id)
                vid.update({
                    "transcript": transcript_text,
                    "char_count": len(transcript_text),
                    "ingestion_engine": engine,
                    "status": "READY",  # آماده برای پردازش در لایه دوم
                    "error_log": firestore.DELETE_FIELD,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                })
                doc_ref.set(vid, merge=True)
                logger.info(f"[SUCCESS] Stored {vid_id} ({len(transcript_text):,} chars via {engine}) in Firestore as READY.")
            except Exception as e:
                logger.error(f"[FAILED] Could not extract transcript for {vid_id}: {e}")
                vid.update({
                    "status": "INGESTION_FAILED",
                    "error_log": str(e),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                })
                doc_ref.set(vid, merge=True)

    def cleanup(self):
        if self.cookie_file_path and os.path.exists(self.cookie_file_path):
            os.remove(self.cookie_file_path)


if __name__ == "__main__":
    pipeline = EduExtractIngestionPipeline()
    
    # 1. خواندن موضوع از داشبورد وب
    target_topic, top_count = pipeline.get_remote_config()
    
    try:
        # 2. اجرای استخراج
        pipeline.run(topic=target_topic, top_k=top_count)
    finally:
        pipeline.cleanup()