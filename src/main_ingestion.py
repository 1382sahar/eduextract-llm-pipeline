import os
import re
import json
import base64
import logging
import tempfile
import urllib.request
from http.cookiejar import MozillaCookieJar
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
    Unified Layer 0 (Discovery via YouTube API v3) & Layer 1 (Authenticated Multi-Engine Transcript Ingestion)
    integrated with Firebase Firestore state management.
    """

    SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

    PIPED_INSTANCES = [
        "https://pipedapi.kavin.rocks",
        "https://pipedapi.tokhmi.xyz",
        "https://pipedapi.syncpundit.io",
        "https://api.piped.private.coffee"
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
            logger.warning("YOUTUBE_COOKIES_B64 not set. Proceeding without cookies.")
            return None

        try:
            decoded = base64.b64decode(self.cookies_b64).decode("utf-8")
            temp_cookie = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8")
            # Ensure Netscape header exists for MozillaCookieJar compatibility
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
        """Creates a requests.Session pre-loaded with the Netscape cookie file and browser headers."""
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

    def _fetch_primary_transcript(self, video_id: str) -> str:
        """Engine 1: youtube-transcript-api v1.0+ WITH authenticated cookie session."""
        session = self._build_authenticated_session()
        try:
            ytt_api = YouTubeTranscriptApi(http_client=session)
            fetched = ytt_api.fetch(video_id, languages=self.languages)
            return " ".join([
                (seg.text if hasattr(seg, "text") else seg["text"]).replace("\n", " ")
                for seg in fetched
            ])
        except TypeError:
            # Fallback if installed version doesn't take http_client
            proxies = {"http": self.proxy_url, "https": self.proxy_url} if self.proxy_url else None
            t_list = YouTubeTranscriptApi.list_transcripts(video_id, cookies=self.cookie_file_path, proxies=proxies)
            transcript = t_list.find_transcript(self.languages)
            return " ".join([seg["text"].replace("\n", " ") for seg in transcript.fetch()])

    def _fetch_ytdlp_fallback(self, video_id: str) -> str:
        """
        Engine 2: yt-dlp WITHOUT conflicting mobile player_client overrides when cookies are present.
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts: Dict[str, Any] = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": self.languages,
            "quiet": True,
            "no_warnings": True,
        }

        if self.cookie_file_path:
            # Web cookies MUST use default web/mweb clients, NOT ios/tv_embedded
            ydl_opts["cookiefile"] = self.cookie_file_path
        else:
            ydl_opts["extractor_args"] = {"youtube": {"player_client": ["ios", "web"]}}

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

            # Fallback to any English variant (e.g., en-US, en-GB, en-orig)
            if not tracks:
                for key in list(subs.keys()) + list(auto_subs.keys()):
                    if key.startswith("en"):
                        tracks = subs.get(key) or auto_subs.get(key)
                        break

            if not tracks:
                raise ValueError(f"No subtitles found in metadata for {video_id}")

            json3_url = next((t["url"] for t in tracks if t.get("ext") == "json3"), tracks[0]["url"])
            session = self._build_authenticated_session()
            resp = session.get(json3_url, timeout=15)
            resp.raise_for_status()

            # Parse either json3 or VTT/XML
            if "json3" in json3_url or resp.text.lstrip().startswith("{"):
                sub_data = resp.json()
                words = [
                    seg.get("utf8", "").strip()
                    for event in sub_data.get("events", [])
                    for seg in event.get("segs", [])
                    if seg.get("utf8", "").strip() and seg.get("utf8") != "\n"
                ]
                return " ".join(words)
            else:
                clean_text = re.sub(r"<[^>]+>", " ", resp.text)
                return " ".join(clean_text.split())

    def _fetch_piped_fallback(self, video_id: str) -> str:
        """Engine 3: Public Piped API fallback for subtitle streams."""
        for instance in self.PIPED_INSTANCES:
            try:
                resp = requests.get(f"{instance}/streams/{video_id}", timeout=10)
                if resp.status_code != 200:
                    continue
                subtitles = resp.json().get("subtitles", [])
                sub_url = None
                for sub in subtitles:
                    if sub.get("code", "").startswith("en"):
                        sub_url = sub.get("url")
                        break
                if sub_url:
                    sub_resp = requests.get(sub_url, timeout=10)
                    if sub_resp.status_code == 200:
                        lines = [
                            re.sub(r"<[^>]+>", "", l.strip())
                            for l in sub_resp.text.splitlines()
                            if l.strip() and not l.startswith("WEBVTT") and "-->" not in l and not l.strip().isdigit()
                        ]
                        text = " ".join(lines)
                        if len(text) > 100:
                            return text
            except Exception:
                continue
        raise RuntimeError(f"All Piped instances failed for {video_id}")

    def extract_transcript(self, video_id: str) -> tuple[str, str]:
        """Tries Engine 1 -> Engine 2 -> Engine 3 and logs exact exception reason."""
        try:
            return self._fetch_primary_transcript(video_id), "youtube-transcript-api"
        except Exception as e1:
            logger.warning(f"Engine 1 failed for {video_id} ({type(e1).__name__}: {e1}). Trying Engine 2 (yt-dlp)...")
            try:
                return self._fetch_ytdlp_fallback(video_id), "yt-dlp-authenticated"
            except Exception as e2:
                logger.warning(f"Engine 2 failed for {video_id} ({type(e2).__name__}: {e2}). Trying Engine 3 (Piped)...")
                return self._fetch_piped_fallback(video_id), "piped-mirror"

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