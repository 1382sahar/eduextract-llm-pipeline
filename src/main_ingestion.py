import os
import re
import glob
import base64
import logging
import tempfile
import subprocess
from http.cookiejar import MozillaCookieJar
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import requests
import yt_dlp

from d1_client import D1PipelineClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class EduExtractIngestionPipeline:
    """
    Unified Layer 0 & Layer 1:
    - Supports general topic discovery, exact video title lookup, and direct YouTube URLs/IDs.
    - Extracts transcripts via dual yt-dlp engines (Python API + CLI VTT fallback).
    - Persists pipeline state and topic queue updates directly to Cloudflare D1.
    """

    SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
    YT_URL_REGEX = re.compile(
        r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|embed/|shorts/|live/)|youtu\.be/)([A-Za-z0-9_-]{11})"
    )
    MAX_TRANSCRIPT_CHARS = 450_000  # Safety cap well below Cloudflare D1's 1MB row limit

    def __init__(self, languages: List[str] = ["en", "fa"]):
        self.languages = languages
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        self.cookies_b64 = os.getenv("YOUTUBE_COOKIES_B64")
        self.proxy_url = os.getenv("YOUTUBE_PROXY_URL")

        if not self.youtube_api_key:
            raise ValueError("Missing YOUTUBE_API_KEY environment variable!")

        self.d1 = D1PipelineClient()
        self.cookie_file_path = self._prepare_cookie_file()

    def get_target_request(self) -> Tuple[str, int, Optional[Any]]:
        """
        Determines target topic / video title / YouTube URL:
        1. If workflow_dispatch provided TARGET_TOPIC, uses it and links any matching PENDING queue row.
        2. Otherwise pops the oldest PENDING request from D1 topic_queue.
        3. Falls back to scheduled default if queue is empty.
        """
        env_topic = (os.getenv("TARGET_TOPIC") or "").strip()
        env_k = (os.getenv("TOP_K_VIDEOS") or "").strip()
        pending_queue = self.d1.get_pending_queue()

        if env_topic:
            top_k = min(max(int(env_k or 3), 1), 10)
            matched_id = None
            for item in pending_queue:
                if str(item.get("topic", "")).strip().lower() == env_topic.lower():
                    matched_id = item.get("id")
                    break
            logger.info(
                f"[CONFIG] Using workflow input -> Target: '{env_topic}', Count: {top_k} "
                f"(queue_id={matched_id})"
            )
            return env_topic, top_k, matched_id

        if pending_queue:
            first = pending_queue[0]
            topic = str(first.get("topic", "")).strip()
            top_k = min(max(int(first.get("video_count") or 3), 1), 10)
            queue_id = first.get("id")
            logger.info(f"[CONFIG] Loaded from D1 topic_queue (#{queue_id}) -> Target: '{topic}', Count: {top_k}")
            return topic, top_k, queue_id

        fallback_topic = "Linear Algebra"
        fallback_k = 2
        logger.info(f"[CONFIG] Queue empty; using scheduled default -> Topic: '{fallback_topic}', Count: {fallback_k}")
        return fallback_topic, fallback_k, None

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
            "Accept-Language": "en-US,en;q=0.9,fa;q=0.8",
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

    def _extract_direct_video_id(self, user_input: str) -> Optional[str]:
        cleaned = user_input.strip()
        match = self.YT_URL_REGEX.search(cleaned)
        if match:
            return match.group(1)
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", cleaned):
            return cleaned
        return None

    @staticmethod
    def _is_specific_title_query(query: str, top_k: int) -> bool:
        """
        Detects when the user supplied a specific video title (e.g., exact lecture name,
        quoted string, non-ASCII/Persian title, or single-video search) rather than a broad topic.
        """
        q = query.strip()
        if top_k == 1:
            return True
        if any(ch in q for ch in ('"', "'", "|", ":", " - ", "—")):
            return True
        if any(ord(ch) > 127 for ch in q):  # Persian or other non-English title/topic
            return True
        if len(q.split()) >= 5:
            return True
        return False

    def fetch_single_video_metadata(self, video_id: str) -> List[Dict[str, Any]]:
        logger.info(f"[DIRECT VIDEO] Fetching metadata for YouTube Video ID: {video_id}")
        stats_params = {
            "part": "snippet,statistics,contentDetails",
            "id": video_id,
            "key": self.youtube_api_key,
        }
        resp = requests.get(self.VIDEOS_URL, params=stats_params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("items", [])

        if not items:
            return [{
                "video_id": video_id,
                "title": f"YouTube Video ({video_id})",
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "topic": "Direct Video",
            }]

        item = items[0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        title = snippet.get("title") or f"YouTube Video ({video_id})"

        return [{
            "video_id": video_id,
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "channel": snippet.get("channelTitle"),
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)),
            "duration_minutes": self._parse_iso_duration(item.get("contentDetails", {}).get("duration", "")),
            "topic": title[:80],
        }]

    def _search_youtube_ids(self, query: str, specific_mode: bool, require_cc: bool) -> List[str]:
        search_params: Dict[str, Any] = {
            "part": "snippet",
            "q": query if specific_mode else f"{query} lecture tutorial explained",
            "type": "video",
            "order": "relevance" if specific_mode else "viewCount",
            "maxResults": 10 if specific_mode else 20,
            "key": self.youtube_api_key,
        }
        if require_cc:
            search_params["videoCaption"] = "closedCaption"
        if not specific_mode:
            search_params["relevanceLanguage"] = "en"

        resp = requests.get(self.SEARCH_URL, params=search_params, timeout=15)
        resp.raise_for_status()
        return [
            item["id"]["videoId"]
            for item in resp.json().get("items", [])
            if "videoId" in item.get("id", {})
        ]

    def discover_top_videos(self, topic: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 1. Direct YouTube URL or 11-character Video ID
        direct_id = self._extract_direct_video_id(topic)
        if direct_id:
            return self.fetch_single_video_metadata(direct_id)

        # 2. Specific Video Title vs Broad Educational Topic
        specific_mode = self._is_specific_title_query(topic, top_k)
        clean_query = topic.strip().strip('"').strip("'")
        logger.info(
            f"Searching YouTube API v3 ({'EXACT TITLE' if specific_mode else 'TOPIC DISCOVERY'}): '{clean_query}'..."
        )

        # In specific title mode, don't filter out videos that only have auto-generated captions
        video_ids = self._search_youtube_ids(clean_query, specific_mode=specific_mode, require_cc=not specific_mode)
        if not video_ids:
            logger.info("No closed-caption results found; retrying broad relevance search...")
            video_ids = self._search_youtube_ids(clean_query, specific_mode=True, require_cc=False)

        if not video_ids:
            return []

        stats_params = {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(video_ids),
            "key": self.youtube_api_key,
        }
        stats_resp = requests.get(self.VIDEOS_URL, params=stats_params, timeout=15)
        stats_resp.raise_for_status()

        items_by_id = {item["id"]: item for item in stats_resp.json().get("items", [])}
        min_duration = 1.0 if specific_mode else 4.0

        candidates = []
        for rank_idx, vid_id in enumerate(video_ids):
            item = items_by_id.get(vid_id)
            if not item:
                continue

            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            duration_min = self._parse_iso_duration(item.get("contentDetails", {}).get("duration", ""))

            if duration_min < min_duration:
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
                "relevance_rank": rank_idx,
                "topic": clean_query[:80],
            })

        if specific_mode:
            # Preserve YouTube's relevance ranking so exact video titles come first
            return candidates[:top_k]

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
            "subtitleslangs": ["en.*", "en", "fa.*", "fa"],
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
            for lang in self.languages:
                for key in subs:
                    if key == lang or key.startswith(f"{lang}-"):
                        tracks = subs[key]
                        break
                if tracks:
                    break

            if not tracks:
                for lang in self.languages:
                    for key in auto_subs:
                        if key == lang or key.startswith(f"{lang}-"):
                            tracks = auto_subs[key]
                            break
                    if tracks:
                        break

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
            return text[: self.MAX_TRANSCRIPT_CHARS]

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
                "--sub-langs", "en.*,en,fa.*,fa",
                "--sub-format", "vtt",
                "-o", out_template,
                url,
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
            return full_text[: self.MAX_TRANSCRIPT_CHARS]

    def extract_transcript(self, video_id: str) -> Tuple[str, str]:
        """Tries Engine 1 (yt-dlp Python API) -> Engine 2 (yt-dlp CLI Disk Writer)."""
        try:
            return self._fetch_ytdlp_python(video_id), "yt-dlp-python"
        except Exception as e1:
            logger.warning(f"Engine 1 failed for {video_id} ({type(e1).__name__}: {e1}). Trying Engine 2 (yt-dlp CLI)...")
            return self._fetch_ytdlp_cli_fallback(video_id), "yt-dlp-cli"

    def run(self, topic: str, top_k: int = 5) -> int:
        existing_videos = {v["id"]: v.get("status", "UNKNOWN") for v in self.d1.get_videos()}
        top_videos = self.discover_top_videos(topic=topic, top_k=top_k)

        if not top_videos:
            logger.warning(f"No matching YouTube videos found for '{topic}'.")
            return 0

        success_count = 0
        for vid in top_videos:
            vid_id = vid["video_id"]
            current_status = existing_videos.get(vid_id)

            # SYSTEM STATE LOCK: preserve videos already ingested or processed
            if current_status in ["READY", "INGESTED", "EXTRACTED", "CLEANED", "SYNTHESIS_FAILED", "COMPLETED"]:
                logger.info(f"[SKIP] Video {vid_id} already in D1 state: '{current_status}'. Preserving lifecycle.")
                success_count += 1
                continue

            logger.info(f"Extracting transcript for: {vid['title']} ({vid_id})...")
            try:
                transcript_text, engine = self.extract_transcript(vid_id)
                self.d1.upsert_video({
                    "id": vid_id,
                    "title": vid["title"] or vid_id,
                    "topic": vid.get("topic") or topic[:80],
                    "status": "READY",
                    "raw_transcript": transcript_text,
                    "char_count": len(transcript_text),
                    "error_log": None,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                success_count += 1
                logger.info(f"[SUCCESS] Stored {vid_id} ({len(transcript_text):,} chars via {engine}) in D1 as READY.")
            except Exception as e:
                logger.error(f"[FAILED] Could not extract transcript for {vid_id}: {e}")
                self.d1.upsert_video({
                    "id": vid_id,
                    "title": vid["title"] or vid_id,
                    "topic": vid.get("topic") or topic[:80],
                    "status": "INGESTION_FAILED",
                    "error_log": str(e)[:1000],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })

        return success_count

    def cleanup(self):
        if self.cookie_file_path and os.path.exists(self.cookie_file_path):
            os.remove(self.cookie_file_path)


if __name__ == "__main__":
    pipeline = EduExtractIngestionPipeline()
    target_topic, top_count, queue_id = pipeline.get_target_request()

    if queue_id is not None:
        pipeline.d1.mark_topic_status(queue_id, "PROCESSING")

    try:
        ingested = pipeline.run(topic=target_topic, top_k=top_count)
        if queue_id is not None:
            pipeline.d1.mark_topic_status(queue_id, "DONE" if ingested > 0 else "FAILED")
    except Exception:
        if queue_id is not None:
            pipeline.d1.mark_topic_status(queue_id, "FAILED")
        raise
    finally:
        pipeline.cleanup()