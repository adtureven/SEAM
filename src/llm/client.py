import json
import os
import re
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    OpenAI = None


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float = 0.0, max_tokens: int = 512,
                 max_retries: int = 3, extra_body: dict | None = None,
        log_path: Path | str | None = None,
        provider: str = "api"):
        if OpenAI is None:
            raise ImportError(
                "LLMClient requires the `openai` package. Install dependencies with "
                "`pip install -r requirements.txt`."
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.provider = provider
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        chat_template_kwargs = self.extra_body.get("chat_template_kwargs", {})
        self.thinking_disabled = (
            self.extra_body.get("enable_thinking") is False
            or chat_template_kwargs.get("enable_thinking") is False
        )
        self.total_calls = 0
        self.total_tokens = 0
        self._call_log: list[dict] = []
        self._log_file = None
        if log_path:
            self._log_file = open(log_path, "a", encoding="utf-8")

    def chat(self, messages: list[dict], temperature: float | None = None,
             max_tokens: int | None = None) -> str:
        max_attempts = self.max_retries * 3
        for attempt in range(max_attempts):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                )
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                t0 = time.time()
                response = self.client.chat.completions.create(**kwargs)
                duration = round(time.time() - t0, 3)
                self.total_calls += 1
                usage = {}
                if response.usage:
                    self.total_tokens += response.usage.total_tokens
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                raw_result = response.choices[0].message.content.strip()
                result = self._strip_thinking(raw_result)

                last_user = ""
                for m in reversed(messages):
                    if m["role"] == "user":
                        last_user = m["content"][:500]
                        break
                entry = {
                    "call_idx": self.total_calls,
                    "duration_s": duration,
                    "provider": self.provider,
                    "endpoint": "<redacted>",
                    "model": self.model,
                    "num_messages": len(messages),
                    "last_user_input": last_user,
                    "response": result,
                    "stripped_thinking": raw_result != result,
                    "usage": usage,
                }
                self._call_log.append(entry)
                if self._log_file:
                    self._log_file.write(
                        json.dumps(entry, ensure_ascii=False) + "\n"
                    )
                    self._log_file.flush()

                return result
            except Exception as e:
                if "429" in str(e):
                    time.sleep(3 + attempt * 2)
                elif attempt < max_attempts - 1:
                    time.sleep(2 ** min(attempt, 4))
                else:
                    raise RuntimeError(f"LLM call failed after {max_attempts} retries: {e}")

    def get_call_log(self) -> list[dict]:
        return self._call_log

    @staticmethod
    def _strip_thinking(text: str) -> str:
        text = text.strip()
        text = re.sub(r"(?is)<think>.*?</think>\s*", "", text).strip()
        return text

    def get_stats(self) -> dict:
        return {"total_calls": self.total_calls, "total_tokens": self.total_tokens}

    def close(self):
        if self._log_file:
            self._log_file.close()
            self._log_file = None


def make_llm_from_config(config: dict, log_path: Path | str | None = None) -> LLMClient:
    llm_cfg = config["llm"]
    provider = llm_cfg.get("provider", "api")
    if provider not in ("api", "local"):
        raise ValueError(f"Unsupported llm.provider: {provider}")
    api_key = llm_cfg.get("api_key")
    if llm_cfg.get("api_key_env"):
        env_name = str(llm_cfg["api_key_env"])
        api_key = os.environ.get(env_name)
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {env_name}")
    elif llm_cfg.get("api_key_file"):
        api_key = Path(llm_cfg["api_key_file"]).expanduser().read_text().strip()
    elif api_key is None:
        api_key = "EMPTY" if provider == "local" else ""

    return LLMClient(
        base_url=llm_cfg["base_url"],
        api_key=api_key,
        model=llm_cfg["model"],
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 512),
        max_retries=llm_cfg.get("max_retries", 3),
        extra_body=llm_cfg.get("extra_body"),
        log_path=log_path,
        provider=provider,
    )
