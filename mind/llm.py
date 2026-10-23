"""LLM backends. Everything else depends only on the `LLM` protocol."""
import json
import urllib.request
from typing import Callable, Protocol


class LLM(Protocol):
    def chat(self, messages: list[dict], *, json_mode: bool = False) -> str: ...


class OllamaClient:
    def __init__(self, model: str, url: str = "http://localhost:11434", temperature: float = 0.8, timeout: float = 300):
        self.model, self.url, self.temperature, self.timeout = model, url.rstrip("/"), temperature, timeout

    def chat(self, messages: list[dict], *, json_mode: bool = False) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            body["format"] = "json"
        req = urllib.request.Request(
            f"{self.url}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)["message"]["content"]


class FakeLLM:
    """Scripted backend for tests: pass a list of replies (popped in order, last one repeats)
    or a callable(messages, json_mode) -> str."""

    def __init__(self, replies: list[str] | Callable[[list[dict], bool], str]):
        self.replies, self.calls = replies, []

    def chat(self, messages: list[dict], *, json_mode: bool = False) -> str:
        self.calls.append(messages)
        if callable(self.replies):
            return self.replies(messages, json_mode)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
