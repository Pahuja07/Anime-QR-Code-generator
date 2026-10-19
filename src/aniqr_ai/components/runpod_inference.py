"""Client for submitting and polling AniQR jobs on RunPod Serverless."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aniqr_ai.exception import AniQRException


class RunPodTimeoutError(AniQRException):
    """Raised when an inference job does not finish within the client deadline."""


class RunPodUnscannableError(AniQRException):
    """Raised when the generated image failed the worker's scan check."""


class RunPodInferenceClient:
    """Submit jobs to a configured RunPod endpoint and poll to a terminal state."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint_id: str | None = None,
        timeout_seconds: float = 300,
        poll_interval_seconds: float = 2,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.getenv("RUNPOD_API_KEY", "")).strip()
        self.endpoint_id = (endpoint_id if endpoint_id is not None else os.getenv("RUNPOD_ENDPOINT_ID", "")).strip()
        if (self.api_key and not self.endpoint_id) or (self.endpoint_id and not self.api_key):
            raise ValueError("RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set together")
        if self.endpoint_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.endpoint_id):
            raise ValueError("RUNPOD_ENDPOINT_ID has an invalid format")
        if timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("RunPod timeout and poll interval must be positive")

        self.enabled = bool(
            self.api_key
            and self.endpoint_id
            and "your_runpod" not in self.api_key.lower()
            and "your_runpod" not in self.endpoint_id.lower()
            and "placeholder" not in self.api_key.lower()
            and "placeholder" not in self.endpoint_id.lower()
        )
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"https://api.runpod.ai/v2/{self.endpoint_id}/{path}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            url,
            data=body,
            method="POST" if body is not None else "GET",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=min(30, self.timeout_seconds)) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise AniQRException(f"RunPod request failed with HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise AniQRException(f"RunPod request failed: {exc}") from exc
        if not isinstance(result, dict):
            raise AniQRException("RunPod returned an invalid response")
        return result

    def generate(self, job_input: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise AniQRException("RunPod inference is not configured")

        deadline = time.monotonic() + self.timeout_seconds
        result = self._request("run", {"input": job_input})
        job_id = result.get("id")
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
            raise AniQRException("RunPod did not return a job id")

        status = result.get("status")
        while status not in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RunPodTimeoutError("RunPod generation timed out")
            time.sleep(min(self.poll_interval_seconds, remaining))
            result = self._request(f"status/{job_id}")
            status = result.get("status")

        if status != "COMPLETED":
            raise AniQRException(f"RunPod job ended with status {status}")
        output = result.get("output")
        if not isinstance(output, dict):
            raise AniQRException("RunPod job completed without an output object")
        if output.get("status") != "COMPLETED":
            if output.get("is_scannable") is False:
                raise RunPodUnscannableError(
                    str(output.get("error") or "Generated QR failed scan validation")
                )
            raise AniQRException(str(output.get("error") or "RunPod inference failed"))
        return output
