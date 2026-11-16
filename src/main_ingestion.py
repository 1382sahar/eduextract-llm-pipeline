import os
import re
import json
import base64
import logging
import tempfile
import urllib.request
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests
import yt_dlp
import firebase_admin
from firebase_admin import credentials, firestore
from youtube_transcript_api import YouTubeTranscriptApi
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class EduExtractIngestionPipeline:
    """
    Unified Layer 0 (Discovery via YouTube API v3) & Layer 1 (Multi-Engine Anti-Bot Transcript Ingestion)
    integrated with Firebase Firestore state management.
    """

    SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

    # Public Invidious instances for cookie-less datacenter fallback
    INVIDIOUS_INSTANCES = [
        "https://inv.tux.pizza",
        "https://invidious.nerdvpn.de",
        "https://invidious.privacyredirect.com",
        "https://iv.ggtyler.dev"
    ]

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

    def _prepare_cookie_file(self) -> Optional[str]:
        if not self.cookies_b64 or not self.cookies_b64.strip():
            logger.warning("YOUTUBE_COOKIES_B64 not set. Proceeding without cookies (multi-fallback active).")
            return None

        try:
            decoded = base64.b64decode(self.cookies_b64).decode("utf-8")
            temp_cookie = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8")
            temp_cookie.write(decoded)
            temp_cookie.close()
            logger.info("Loaded YouTube cookies from YOUTUBE_COOKIES_B64 secret.")
            return temp_cookie.name
        except Exception as e:
            logger.error(f"Cookie decode error: {e}")
            return None

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

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=2, max=6))
    def _fetch_primary_transcript(self, video_id: str) -> str:
        """ Engine 1: Compatible with both youtube-transcript-api v1.0+ and v0.6.x """
        try:
            # New API (v1.0.0+)
            ytt_api = YouTubeTranscriptApi()
            fetched = ytt_api.fetch(video_id, languages=self.languages)
            return " ".join([
                (seg.text if hasattr(seg, "text") else seg["text"]).replace("\n", " ")
                for seg in fetched
            ])
        except (AttributeError, TypeError):
            # Legacy API (v0.6.x)
            proxies = {"http": self.proxy_url, "https": self.proxy_url} if self.proxy_url else None
            t_list = YouTubeTranscriptApi.list_transcripts(video_id, cookies=self.cookie_file_path, proxies=proxies)
            transcript = t_list.find_transcript(self.languages)
            return " ".join([seg["text"].replace("\n", " ") for seg in transcript.fetch()])

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=2, max=8))
    def _fetch_ytdlp_fallback(self, video_id: str) -> str:
        """ Engine 2: yt-dlp with TV/iOS client spoofing & optional cookies """
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts: Dict[str, Any] = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": self.languages,
            "quiet": True,
            "no_warnings": True,
            # tv_embedded and ios bypass most unauthenticated bot checks
            "extractor_args": {"youtube": {"player_client": ["tv_embedded", "ios", "android"]}},
        }
        if self.cookie_file_path:
            ydl_opts["cookiefile"] = self.cookie_file_path
        if self.proxy_url:
            ydl_opts["proxy"] = self.proxy_url

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subs = info.get("subtitles") or {}
            auto_subs = info.get("automatic_captions") or {}

            tracks = None
            for lang in self.languages:
                if lang in subs:
                    tracks = subs[lang]
                    break
                if lang in auto_subs:
                    tracks = auto_subs[lang]
                    break

            if not tracks:
                raise ValueError(f"No subtitles available for {video_id}")

            json3_url = next((t["url"] for t in tracks if t.get("ext") == "json3"), tracks[0]["url"])
            req = urllib.request.Request(json3_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                sub_data = json.loads(resp.read().decode("utf-8"))

            words = [
                seg.get("utf8", "").strip()
                for event in sub_data.get("events", [])
                for seg in event.get("segs", [])
                if seg.get("utf8", "").strip() and seg.get("utf8") != "\n"
            ]
            return " ".join(words)

    def _fetch_invidious_fallback(self, video_id: str) -> str:
        """ Engine 3: Cookie-less fallback via public Invidious instances (bypasses datacenter IP bans) """
        for instance in self.INVIDIOUS_INSTANCES:
            try:
                api_url = f"{instance}/api/v1/captions/{video_id}"
                resp = requests.get(api_url, timeout=10)
                if resp.status_code != 200:
                    continue

                captions = resp.json().get("captions", [])
                target_url = None
                for cap in captions:
                    if cap.get("language_code", "").startswith("en"):
                        target_url = f"{instance}{cap['url']}"
                        break

                if not target_url and captions:
                    target_url = f"{instance}{captions[0]['url']}"

                if target_url:
                    vtt_resp = requests.get(target_url, timeout=10)
                    if vtt_resp.status_code == 200:
                        lines = []
                        for line in vtt_resp.text.splitlines():
                            line = line.strip()
                            if not line or line.startswith("WEBVTT") or "-->" in line or line.isdigit():
                                continue
                            clean_line = re.sub(r"<[^>]+>", "", line)
                            lines.append(clean_line)
                        full_text = " ".join(lines)
                        if len(full_text) > 100:
                            return full_text
            except Exception as e:
                logger.debug(f"Invidious instance {instance} failed: {e}")
                continue
        raise RuntimeError(f"All Invidious instances failed for {video_id}")

    def extract_transcript(self, video_id: str) -> tuple[str, str]:
        """Tries Engine 1 -> Engine 2 -> Engine 3 sequentially."""
        try:
            return self._fetch_primary_transcript(video_id), "youtube-transcript-api"
        except Exception as e1:
            logger.warning(f"Engine 1 failed for {video_id} ({type(e1).__name__}). Trying Engine 2 (yt-dlp)...")
            try:
                return self._fetch_ytdlp_fallback(video_id), "yt-dlp-fallback"
            except Exception as e2:
                logger.warning(f"Engine 2 failed for {video_id}. Trying Engine 3 (Invidious Mirror)...")
                return self._fetch_invidious_fallback(video_id), "invidious-mirror"

    def run(self, topic: str, top_k: int = 5):
        top_videos = self.discover_top_videos(topic=topic, top_k=top_k)
        col_ref = self.db.collection("Pipeline_State")

        for vid in top_videos:
            vid_id = vid["video_id"]
            doc_ref = col_ref.document(vid_id)
            snapshot = doc_ref.get()

            if snapshot.exists and snapshot.to_dict().get("status") in ["INGESTED", "EXTRACTED", "VALIDATED", "SYNCED"]:
                logger.info(f"[SKIP] Video {vid_id} already processed (status: {snapshot.to_dict().get('status')}).")
                continue

            logger.info(f"Extracting transcript for: {vid['title']} ({vid_id})...")
            try:
                transcript_text, engine = self.extract_transcript(vid_id)
                vid.update({
                    "transcript": transcript_text,
                    "char_count": len(transcript_text),
                    "ingestion_engine": engine,
                    "status": "INGESTED",
                    "error_log": firestore.DELETE_FIELD,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                })
                doc_ref.set(vid, merge=True)
                logger.info(f"[SUCCESS] Stored {vid_id} ({len(transcript_text):,} chars via {engine}) in Firestore.")
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
    target_topic = os.getenv("TARGET_TOPIC", "Linear Algebra")
    top_count = int(os.getenv("TOP_K_VIDEOS", "5"))

    pipeline = EduExtractIngestionPipeline()
    try:
        pipeline.run(topic=target_topic, top_k=top_count)
    finally:
        pipeline.cleanup()