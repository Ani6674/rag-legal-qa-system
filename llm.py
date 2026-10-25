"""Unified LLM wrapper: ollama (local), anthropic (API), gemini (API), hf (local transformers)."""
import os

import requests

import config


class LLM:
    def __init__(self, backend: str | None = None):
        self.backend = backend or config.LLM_BACKEND
        self._pipe = None
        self._client = None

    @property
    def name(self) -> str:
        return {
            "ollama": config.OLLAMA_MODEL,
            "anthropic": config.ANTHROPIC_MODEL,
            "gemini": config.GEMINI_MODEL,
            "hf": config.HF_MODEL,
        }[self.backend]

    def generate(self, prompt: str, system: str = "", max_tokens: int = 400,
                 temperature: float = 0.0, json_mode: bool = False) -> str:
        """json_mode forces valid JSON output on Ollama (ignored by other backends)."""
        if self.backend == "ollama":
            return self._ollama(prompt, system, max_tokens, temperature, json_mode).strip()
        fn = getattr(self, f"_{self.backend}")
        return fn(prompt, system, max_tokens, temperature).strip()

    # ---------------- backends ----------------
    def _ollama(self, prompt, system, max_tokens, temperature, json_mode=False):
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        body = {
            "model": config.OLLAMA_MODEL, "messages": messages, "stream": False,
            "keep_alive": "30m",                       # keep the model in memory between calls
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = "json"
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", timeout=600, json=body)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _anthropic(self, prompt, system, max_tokens, temperature):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic()  # reads ANTHROPIC_API_KEY
        kwargs = dict(model=config.ANTHROPIC_MODEL, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        try:
            resp = self._client.messages.create(temperature=temperature, **kwargs)
        except Exception:  # some models reject the temperature parameter
            resp = self._client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def _gemini(self, prompt, system, max_tokens, temperature):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            body["system_instruction"] = {"parts": [{"text": system}]}
        r = requests.post(url, json=body, timeout=120,
                          headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    def _hf(self, prompt, system, max_tokens, temperature):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("text-generation", model=config.HF_MODEL, device_map="auto")
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        out = self._pipe(messages, max_new_tokens=max_tokens, do_sample=temperature > 0,
                         temperature=temperature if temperature > 0 else None)
        return out[0]["generated_text"][-1]["content"]
