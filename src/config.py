"""Model routing in one place.

The roadmap puts this in Phase 6, but the Extractor is the second LLM
node and the first one that runs on an open model, so "which provider,
which model, which key" already has two answers. Env vars stay the
override mechanism; this module is where the defaults live. It holds the
Extractor's and the Ranker's routes; Phase 6 adds the Judge's and (moved
from tailor.py) the Tailor's.

It also reads RANKER_EXCLUDE_ATOMS, the one Ranker setting that is personal
data rather than a model choice, and holds MODEL_SETTINGS: the sampling
and reasoning parameters sent with every request, per model — and, for a
model whose id exists on one provider only, which provider serves it
(PROVIDERS, route_for). `--model qwen-3.8-27b` then goes to Cerebras with
no .env change, while the .env route stays the pipeline's default.

Every provider here speaks the OpenAI chat-completions protocol, so the
Extractor's and Ranker's client is the same
`openai.OpenAI(base_url=..., api_key=...)` for all of them. Switching is a
`.env` change, not a code change (RANKER_* mirrors EXTRACTOR_*):

    provider   EXTRACTOR_BASE_URL                    EXTRACTOR_KEY_ENV
    Groq       https://api.groq.com/openai/v1        GROQ_API_KEY      (default)
    OpenRouter https://openrouter.ai/api/v1          OPENROUTER_API_KEY
    Ollama     http://localhost:11434/v1             (any non-empty value)

Model ids are read from the environment on purpose: Groq's catalogue
changes under you (models get deprecated from the free tier), so the
one true list is `python -m src.agents.extractor --list-models`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# The answer budget every request reserves unless its model says otherwise.
# Reasoning models spend it on thinking before they answer; see MAX_TOKENS
# in src/agents/extractor.py for the gpt-oss history behind 8192.
DEFAULT_MAX_TOKENS = 8192

# Placeholder until you pin one from --list-models. A wrong id fails fast
# with the provider's "model not found" — it never silently degrades.
DEFAULT_EXTRACTOR_MODEL = "openai/gpt-oss-120b"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelRoute:
    base_url: str
    key_env: str      # name of the env var holding the key
    model: str

    def api_key(self) -> str:
        key = os.environ.get(self.key_env)
        if not key:
            raise ConfigError(f"{self.key_env} is not set; add it to .env")
        return key


def extractor_route() -> ModelRoute:
    return ModelRoute(
        base_url=os.environ.get("EXTRACTOR_BASE_URL", GROQ_BASE_URL),
        key_env=os.environ.get("EXTRACTOR_KEY_ENV", "GROQ_API_KEY"),
        model=os.environ.get("EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL),
    )


# The Ranker's stage-2 rerank is the same kind of job — structured output
# from an open model — so it starts on the same model the Extractor was
# accepted on. It has its own RANKER_* variables so a smaller, cheaper
# model can be tried for reranking without touching extraction.
DEFAULT_RANKER_MODEL = DEFAULT_EXTRACTOR_MODEL


def ranker_route() -> ModelRoute:
    return ModelRoute(
        base_url=os.environ.get("RANKER_BASE_URL", GROQ_BASE_URL),
        key_env=os.environ.get("RANKER_KEY_ENV", "GROQ_API_KEY"),
        model=os.environ.get("RANKER_MODEL", DEFAULT_RANKER_MODEL),
    )


def ranker_exclude_atoms(value: str | None = None) -> frozenset[str]:
    """Profile atom ids the Ranker's stage 1 leaves out: `value` if given
    (a CLI flag; "" means none), else RANKER_EXCLUDE_ATOMS. Comma-separated.
    They live in .env, not code, because they name entries of the private
    profile.yaml. An id that isn't in the profile fails in the Ranker."""
    raw = os.environ.get("RANKER_EXCLUDE_ATOMS", "") if value is None else value
    return frozenset(a.strip() for a in raw.split(",") if a.strip())


# ---------------------------------------------------------------------------
# Per-model request settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSettings:
    """What goes into every request besides the prompt. Sent explicitly so
    no run depends on a provider default, which differs per model and can
    change without notice. Values follow Groq's API reference (2026-09):

      reasoning_effort  gpt-oss: low | medium | high (Groq default: medium).
                        qwen/qwen3.8-27b: none | default | low | medium | high
                        (Groq default: none, i.e. no reasoning at all).
                        A value outside a model's set is a 400.
      reasoning_format  Qwen and MiniMax only: raw | parsed | hidden. The
                        default, raw, puts the reasoning inside the answer in
                        <think> tags and is refused with JSON mode. gpt-oss
                        rejects the parameter, so it is only sent when set.
      max_tokens        The answer budget. Groq checks prompt + max_tokens
                        against the model's tokens-per-minute limit *before*
                        running, and refuses a request that could exceed it
                        ("Request too large"), however few tokens the answer
                        would really use. On the free tier that limit is 8K
                        for Qwen 3.8 (2026-09).
    """

    temperature: float = 0.0
    top_p: float | None = None
    reasoning_effort: str | None = None       # None: not sent (provider default)
    reasoning_format: str | None = None       # None: not sent
    max_tokens: int = DEFAULT_MAX_TOKENS
    # A PROVIDERS name when the model id exists on that provider only; None
    # means "whatever the .env route points at" (gpt-oss ids are the same on
    # Groq and OpenRouter, so pinning them would override an .env switch).
    provider: str | None = None

    def request_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for `client.chat.completions.create`.
        reasoning_format is a Groq extension, so it travels in extra_body."""
        kwargs: dict[str, Any] = {"temperature": self.temperature, "max_tokens": self.max_tokens}
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_format is not None:
            kwargs["extra_body"] = {"reasoning_format": self.reasoning_format}
        return kwargs

    @property
    def label(self) -> str:
        """Short form for eval reports: 'effort medium, T 0'."""
        parts = [f"effort {self.reasoning_effort or 'provider default'}"]
        if self.reasoning_format:
            parts.append(f"reasoning {self.reasoning_format}")
        parts.append(f"T {self.temperature:g}" + (f", top_p {self.top_p:g}" if self.top_p is not None else ""))
        if self.max_tokens != DEFAULT_MAX_TOKENS:
            parts.append(f"max_tokens {self.max_tokens}")
        return ", ".join(parts)


MODEL_SETTINGS: dict[str, ModelSettings] = {
    # Groq's default effort is already medium, so sending it changes nothing
    # for the accepted Extractor (M7, M8) or Ranker runs; it makes it visible.
    # Temperature 0 is what those runs used.
    "openai/gpt-oss-120b": ModelSettings(temperature=0.0, reasoning_effort="medium"),
    "openai/gpt-oss-20b": ModelSettings(temperature=0.0, reasoning_effort="medium"),
    # Groq's default for this model is "none": reasoning off. Unset, the
    # comparison would pit gpt-oss thinking against Qwen not thinking.
    # "hidden": JSON mode refuses the default "raw". Sampling is Groq's
    # recommended thinking-mode setting: Qwen's model cards warn that greedy
    # decoding (temperature 0) can loop while thinking. max_tokens 4096:
    # with 8192, a ~2.9k-token prompt was refused as too large for the 8K
    # per-minute limit (m9, 2026-09-30). Its answers used ~1.3k tokens.
    "qwen/qwen3.8-27b": ModelSettings(temperature=1.0, top_p=0.95, reasoning_effort="medium",
                                      reasoning_format="hidden", max_tokens=4096),
    # The same two models on Cerebras, whose free tier has 30K tokens a
    # minute (Groq's refused Qwen at "limit 1000", m9). Cerebras's defaults
    # differ from Groq's — Qwen reasons at "high" unless told otherwise — so
    # both are pinned to the Groq plan's settings: medium effort, gpt-oss
    # greedy, Qwen at its thinking-mode sampling, the same 8192 budget.
    # "parsed" keeps Qwen's reasoning in its own field, out of the JSON.
    "gpt-oss-120b": ModelSettings(temperature=0.0, reasoning_effort="medium", provider="cerebras"),
    "qwen-3.8-27b": ModelSettings(temperature=1.0, top_p=0.95, reasoning_effort="medium",
                                  reasoning_format="parsed", provider="cerebras"),
}

# name -> (base URL, env var holding the key). All speak the OpenAI
# chat-completions protocol, so one client class serves them all.
PROVIDERS: dict[str, tuple[str, str]] = {
    "groq": (GROQ_BASE_URL, "GROQ_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    # Hugging Face's router; pick the serving provider with a suffix on the
    # model id, e.g. "Qwen/Qwen3.8-27B:novita" or ":cheapest".
    "huggingface": ("https://router.huggingface.co/v1", "HF_TOKEN"),
}


def route_for(model: str, default: ModelRoute, provider: str | None = None) -> ModelRoute:
    """Where a request for `model` goes: `provider` if given (a --provider
    flag), else the provider MODEL_SETTINGS pins the model to, else
    `default` — the .env route — with this model."""
    name = provider or MODEL_SETTINGS.get(model, ModelSettings()).provider
    if name is None:
        return replace(default, model=model)
    if name not in PROVIDERS:
        raise ConfigError(f"unknown provider {name!r}; one of {sorted(PROVIDERS)}")
    base_url, key_env = PROVIDERS[name]
    return ModelRoute(base_url=base_url, key_env=key_env, model=model)


def model_settings(model: str, reasoning_effort: str | None = None) -> ModelSettings:
    """The settings for `model`: MODEL_SETTINGS if listed, else temperature 0
    and provider defaults for everything else (a model tried ad hoc still
    works). `reasoning_effort`, if given, overrides the table (a CLI flag)."""
    settings = MODEL_SETTINGS.get(model, ModelSettings())
    return replace(settings, reasoning_effort=reasoning_effort) if reasoning_effort else settings
