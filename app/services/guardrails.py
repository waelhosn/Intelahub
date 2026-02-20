import re
from dataclasses import dataclass

from app.core.errors import APIError
from app.core.settings import Settings
from app.models.query import QueryRequest


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    code: str | None = None


class DeterministicGuardrails:
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"(ignore|disregard|override).*(system|developer|previous).*(instruction|prompt)", re.IGNORECASE),
        re.compile(r"(do\s+not|don't)\s+follow\s+.*(instructions|policy|guardrails)", re.IGNORECASE),
        re.compile(r"(reveal|show|print|expose).*(api\s*key|secret|token|password)", re.IGNORECASE),
        re.compile(r"(system|developer)\s+prompt", re.IGNORECASE),
        re.compile(r"(show|print|dump).*(environment|env|\.env)", re.IGNORECASE),
        re.compile(r"bypass\s+(safety|guardrails|policy)", re.IGNORECASE),
    ]

    EXFIL_KEYWORDS = {
        "api key",
        "secret",
        "token",
        "password",
        "private key",
    }
    SUSPICIOUS_TOKEN_PATTERNS = [
        re.compile(r"\b(base64|hex)\b.{0,40}\b(decode|dump|print|reveal|env|api[_\s-]*key|secret)\b", re.IGNORECASE),
        re.compile(r"\b(openai[_\s-]*api[_\s-]*key|tmdb[_\s-]*api[_\s-]*key)\b", re.IGNORECASE),
        re.compile(r"\b(os\.environ|getenv|printenv|dotenv|\.env)\b", re.IGNORECASE),
        re.compile(r"\benv\s*\|", re.IGNORECASE),
    ]
    EXFIL_VERBS = {"show", "reveal", "return", "print", "expose", "dump", "decode"}

    def __init__(self, settings: Settings):
        self.settings = settings

    def _looks_like_obfuscated_exfil(self, normalized: str) -> bool:
        has_secret_hint = any(
            token in normalized
            for token in [
                "api key",
                "openai_api_key",
                "tmdb_api_key",
                ".env",
                "secret",
                "token",
                "password",
                "private key",
            ]
        )
        has_exfil_verb = any(verb in normalized for verb in self.EXFIL_VERBS)
        separator_count = len(re.findall(r"[|;&$`><]", normalized))
        return has_secret_hint and has_exfil_verb and separator_count >= 4

    def inspect_text(self, text: str) -> GuardrailResult:
        normalized = re.sub(r"\s+", " ", text.strip().lower())

        if not normalized:
            return GuardrailResult(False, "Query is empty", "invalid_query")

        if len(normalized) > self.settings.max_query_chars:
            return GuardrailResult(False, "Query is too long", "query_too_long")

        for pattern in self.SUSPICIOUS_TOKEN_PATTERNS:
            if pattern.search(normalized):
                return GuardrailResult(False, "Suspicious token pattern detected", "suspicious_token_pattern")

        if self._looks_like_obfuscated_exfil(normalized):
            return GuardrailResult(False, "Suspicious token pattern detected", "suspicious_token_pattern")

        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(normalized):
                return GuardrailResult(False, "Potential prompt injection detected", "prompt_injection")

        if any(keyword in normalized for keyword in self.EXFIL_KEYWORDS) and any(verb in normalized for verb in self.EXFIL_VERBS):
            return GuardrailResult(False, "Potential secret exfiltration detected", "secret_exfiltration")

        return GuardrailResult(True)

    def validate_request(self, request: QueryRequest) -> None:
        inspection = self.inspect_text(request.query)
        if not inspection.allowed:
            raise APIError(
                code=inspection.code or "guardrail_blocked",
                message=inspection.reason or "Query blocked by guardrails",
                status_code=400,
            )

        if request.filters and request.filters.min_year and request.filters.max_year:
            if request.filters.min_year > request.filters.max_year:
                raise APIError(
                    code="invalid_filters",
                    message="min_year cannot be greater than max_year",
                    status_code=400,
                )
