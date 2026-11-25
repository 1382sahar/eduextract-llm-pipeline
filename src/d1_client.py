import os
import time
import logging
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)


class D1PipelineClient:
    """
    Lightweight HTTP client connecting GitHub Actions Python layers (Layer 1-4)
    to the Cloudflare Worker + D1 SQLite database using X-Pipeline-Secret auth.
    """

    def __init__(
        self,
        worker_url: Optional[str] = None,
        pipeline_secret: Optional[str] = None,
        timeout: int = 45,
        max_retries: int = 3,
    ) -> None:
        raw_url = (
            worker_url
            or os.environ.get("CLOUDFLARE_WORKER_URL")
            or "https://eduextract-chat.ssa210819.workers.dev"
        ).strip().rstrip("/")

        for suffix in ("/api/chat", "/api/videos", "/embed", "/health"):
            if raw_url.endswith(suffix):
                raw_url = raw_url[: -len(suffix)]

        self.base_url = raw_url
        self.secret = (pipeline_secret or os.environ.get("PIPELINE_SECRET") or "").strip()
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.secret:
            raise RuntimeError("Missing PIPELINE_SECRET environment variable!")

        self.session = requests.Session()
        self.session.headers.update({
            "X-Pipeline-Secret": self.secret,
            "Content-Type": "application/json",
            "User-Agent": "EduExtract-GitHub-Actions/3.0",
        })

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs,
                )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    wait = attempt * 2
                    logger.warning(f"D1 API transient HTTP {resp.status_code} on {path}; retrying in {wait}s...")
                    time.sleep(wait)
                    continue

                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"D1 API error [{method} {path}] (HTTP {resp.status_code}): {resp.text[:500]}"
                    )
                return resp.json() if resp.text else {}
            except requests.RequestException as exc:
                last_err = exc
                if attempt < self.max_retries:
                    wait = attempt * 2
                    logger.warning(f"D1 API network error on {path} ({exc}); retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"D1 API connection failed [{method} {path}]: {exc}") from exc

        raise RuntimeError(f"D1 API failed after {self.max_retries} attempts: {last_err}")

    def get_pending_queue(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "/api/pipeline/queue")
        return data.get("queue") or []

    def mark_topic_status(self, topic_id: Any, status: str = "DONE") -> None:
        self._request(
            "POST",
            "/api/pipeline/queue/update",
            json={"id": topic_id, "status": status},
        )

    def upsert_video(self, video_payload: Dict[str, Any]) -> None:
        self._request("POST", "/api/pipeline/videos/upsert", json=video_payload)

    def get_videos(
        self,
        status: Optional[str] = None,
        rag_indexed: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {}
        if status is not None:
            params["status"] = status
        if rag_indexed is not None:
            params["rag_indexed"] = "1" if rag_indexed else "0"
        data = self._request("GET", "/api/pipeline/videos", params=params)
        return data.get("videos") or []

    def get_videos_by_statuses(self, statuses: List[str]) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for st in statuses:
            for video in self.get_videos(status=st):
                vid_id = video.get("id")
                if vid_id and vid_id not in seen:
                    seen[vid_id] = video
        return list(seen.values())

    def update_video(self, video_id: str, fields: Dict[str, Any]) -> None:
        self._request(
            "POST",
            "/api/pipeline/videos/update",
            json={"id": video_id, "fields": fields},
        )

    def upsert_vectors(self, vectors: List[Dict[str, Any]], batch_size: int = 10) -> None:
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._request("POST", "/api/pipeline/vectors/upsert", json={"vectors": batch})