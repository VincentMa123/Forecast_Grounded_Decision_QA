"""Provider-specific chat client configuration for the agent orchestrator."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


def _optional_float(environ: Mapping[str, str], name: str) -> Optional[float]:
    value = environ.get(name)
    if value in (None, "", "0"):
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _required_int(environ: Mapping[str, str], name: str, default: int) -> int:
    value = environ.get(name)
    if value in (None, ""):
        return default
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _required_float(environ: Mapping[str, str], name: str, default: float) -> float:
    value = environ.get(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _optional_bool(environ: Mapping[str, str], name: str) -> Optional[bool]:
    value = environ.get(name)
    if value in (None, ""):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError(f"{name} must be true or false.")


@dataclass(frozen=True)
class LLMProviderSettings:
    provider: str
    api_key: Optional[str] = field(repr=False)
    base_url: Optional[str]
    model: str
    timeout_seconds: Optional[float]
    temperature: float
    max_tokens: Optional[int] = None
    thinking_type: Optional[str] = None
    thinking_enabled: Optional[bool] = None
    reasoning_effort: Optional[str] = None

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "LLMProviderSettings":
        values = os.environ if environ is None else environ
        provider = str(values.get("LLM_PROVIDER") or "openai").strip().lower()
        if provider not in {"openai", "zai"}:
            raise ValueError("LLM_PROVIDER must be either 'openai' or 'zai'.")

        if provider == "openai":
            return cls(
                provider=provider,
                api_key=values.get("OPENAI_API_KEY"),
                base_url=values.get("OPENAI_API_BASE"),
                model=str(values.get("OPENAI_MODEL") or "gpt-4"),
                timeout_seconds=_optional_float(
                    values,
                    "OPENAI_TIMEOUT_SECONDS",
                ),
                temperature=0.2,
                reasoning_effort=(
                    str(values["OPENAI_REASONING_EFFORT"]).strip().lower()
                    if values.get("OPENAI_REASONING_EFFORT")
                    else None
                ),
            )

        thinking_type = str(values.get("ZAI_THINKING") or "enabled").strip().lower()
        if thinking_type not in {"enabled", "disabled"}:
            raise ValueError("ZAI_THINKING must be either 'enabled' or 'disabled'.")
        return cls(
            provider=provider,
            api_key=values.get("ZAI_API_KEY"),
            base_url=values.get("ZAI_BASE_URL"),
            model=str(values.get("ZAI_MODEL") or "glm-5.2"),
            timeout_seconds=_optional_float(values, "ZAI_TIMEOUT_SECONDS"),
            temperature=_required_float(values, "ZAI_TEMPERATURE", 1.0),
            max_tokens=_required_int(values, "ZAI_MAX_TOKENS", 65536),
            thinking_type=thinking_type,
            reasoning_effort=str(values.get("ZAI_REASONING_EFFORT") or "max"),
        )


class LLMProvider:
    def __init__(self, settings: LLMProviderSettings) -> None:
        self.settings = settings

    def request_options(self) -> dict[str, Any]:
        if self.settings.provider == "openai":
            options: dict[str, Any] = {
                "temperature": self.settings.temperature,
            }

            if self.settings.reasoning_effort:
                options["reasoning_effort"] = self.settings.reasoning_effort
            if self.settings.max_tokens is not None:
                options["max_tokens"] = self.settings.max_tokens

            return options

        return {
            "thinking": {"type": self.settings.thinking_type},
            "reasoning_effort": self.settings.reasoning_effort,
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
        }

    def create_client(self) -> Any:
        kwargs: dict[str, Any] = {"api_key": self.settings.api_key}
        if self.settings.base_url:
            kwargs["base_url"] = self.settings.base_url
        if self.settings.timeout_seconds is not None:
            kwargs["timeout"] = self.settings.timeout_seconds
        if self.settings.provider == "openai":
            from openai import OpenAI
            
            kwargs["max_retries"] = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
            
            return OpenAI(**kwargs)
        try:
            from zai import ZhipuAiClient
        except ImportError as exc:
            raise RuntimeError(
                "Install zai-sdk from pipeclaw/backend/requirements.txt to use LLM_PROVIDER=zai."
            ) from exc
        return ZhipuAiClient(**kwargs)

    def assistant_history_fields(self, message: Any) -> dict[str, Any]:
        if self.settings.provider != "zai":
            return {}
        reasoning_content = getattr(message, "reasoning_content", None)
        return {"reasoning_content": reasoning_content} if reasoning_content else {}

    def missing_key_message(self) -> Optional[str]:
        if self.settings.api_key:
            return None
        if self.settings.provider == "zai":
            return "Agent service is missing ZAI_API_KEY; cannot run a live GLM call."
        return "Agent service is missing OPENAI_API_KEY; cannot run a live model call."
