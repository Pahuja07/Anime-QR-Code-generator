import pytest

from aniqr_ai.components.runpod_inference import (
    RunPodInferenceClient,
    RunPodTimeoutError,
)


def test_runpod_client_submits_and_polls_job(monkeypatch):
    client = RunPodInferenceClient(
        api_key="test-token",
        endpoint_id="endpoint_123",
        timeout_seconds=10,
        poll_interval_seconds=0.001,
    )
    responses = iter(
        [
            {"id": "job_123", "status": "IN_QUEUE"},
            {
                "id": "job_123",
                "status": "COMPLETED",
                "output": {"status": "COMPLETED", "is_scannable": True},
            },
        ]
    )
    paths = []

    def fake_request(path, payload=None):
        paths.append((path, payload))
        return next(responses)

    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr("aniqr_ai.components.runpod_inference.time.sleep", lambda _: None)

    output = client.generate({"content": "hello", "content_type": "text"})

    assert output["is_scannable"]
    assert paths == [
        ("run", {"input": {"content": "hello", "content_type": "text"}}),
        ("status/job_123", None),
    ]


def test_runpod_client_rejects_partial_configuration():
    with pytest.raises(ValueError, match="must be set together"):
        RunPodInferenceClient(api_key="test-token", endpoint_id="")


def test_runpod_client_times_out(monkeypatch):
    client = RunPodInferenceClient(
        api_key="test-token",
        endpoint_id="endpoint_123",
        timeout_seconds=0.000001,
        poll_interval_seconds=0.001,
    )
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: {"id": "job_123", "status": "IN_QUEUE"})

    with pytest.raises(RunPodTimeoutError, match="timed out"):
        client.generate({"content": "hello"})

