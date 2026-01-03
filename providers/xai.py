"""X.AI (GROK) model provider implementation."""

import logging
import re
from datetime import date
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from utils.env import get_env

from .openai_compatible import OpenAICompatibleProvider
from .registries.xai import XAIModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ModelResponse, ProviderType

logger = logging.getLogger(__name__)


class XAIModelProvider(RegistryBackedProviderMixin, OpenAICompatibleProvider):
    """Integration for X.AI's GROK models exposed over an OpenAI-style API.

    Publishes capability metadata for the officially supported deployments and
    maps tool-category preferences to the appropriate GROK model.
    """

    FRIENDLY_NAME = "X.AI"

    REGISTRY_CLASS = XAIModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    # Canonical model identifiers used for category routing.
    PRIMARY_MODEL = "grok-4-1-fast-reasoning"
    FALLBACK_MODEL = "grok-4"

    _X_POST_URL_PATTERN = re.compile(
        r"https?://(?:www\\.)?(?:x\\.com|twitter\\.com)/[^\\s/]+/(?:status|statuses)/\\d+",
        re.IGNORECASE,
    )

    def __init__(self, api_key: str, **kwargs):
        """Initialize X.AI provider with API key."""
        # Set X.AI base URL
        kwargs.setdefault("base_url", "https://api.x.ai/v1")
        self._ensure_registry()
        super().__init__(api_key, **kwargs)
        self._invalidate_capability_cache()

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.XAI

    def _x_search_enabled(self) -> bool:
        """Return True when xAI's agentic search tooling should be enabled for Grok requests."""

        raw = (get_env("XAI_ENABLE_X_SEARCH", "true") or "").strip().lower()
        return raw in {"1", "true", "yes", "y", "on"}

    def _x_search_tool_choice(self) -> str:
        raw = (get_env("XAI_X_SEARCH_TOOL_CHOICE", "auto") or "").strip().lower()
        return raw if raw in {"auto", "required"} else "auto"

    def _prompt_mentions_x_post(self, prompt: str) -> bool:
        """Detect whether a prompt refers to an X/Twitter post URL."""

        if not prompt:
            return False

        if self._X_POST_URL_PATTERN.search(prompt):
            return True

        # Broader detection for prompts that include an X link but not the canonical status form.
        lowered = prompt.lower()
        return "x.com/" in lowered or "twitter.com/" in lowered

    def _x_search_sources(self) -> list[str]:
        """Return x_search sources per xAI Search Tools docs.

        Defaults to X-only when not configured since this hook is geared toward X URLs.
        Supported values include: x, web, news, rss.
        """

        raw = (get_env("XAI_X_SEARCH_SOURCES", "x") or "").strip()
        if not raw:
            return ["x"]

        sources = [entry.strip().lower() for entry in raw.split(",") if entry.strip()]
        return sources or ["x"]

    def _parse_handle_list_env(self, name: str) -> Optional[list[str]]:
        raw = (get_env(name) or "").strip()
        if not raw:
            return None
        handles = [entry.strip().lstrip("@") for entry in raw.split(",") if entry.strip()]
        return handles or None

    def _parse_date_env(self, name: str) -> Optional[str]:
        raw = (get_env(name) or "").strip()
        if not raw:
            return None
        try:
            date.fromisoformat(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s (expected YYYY-MM-DD): %s", name, raw)
            return None
        return raw

    def _build_x_search_tool(self) -> dict:
        tool: dict = {"type": "x_search"}

        allowed = self._parse_handle_list_env("XAI_X_SEARCH_ALLOWED_HANDLES")
        if allowed:
            tool["allowed_x_handles"] = allowed

        excluded = self._parse_handle_list_env("XAI_X_SEARCH_EXCLUDED_HANDLES")
        if excluded:
            tool["excluded_x_handles"] = excluded

        from_date = self._parse_date_env("XAI_X_SEARCH_FROM_DATE")
        if from_date:
            tool["from_date"] = from_date

        to_date = self._parse_date_env("XAI_X_SEARCH_TO_DATE")
        if to_date:
            tool["to_date"] = to_date

        sources = self._x_search_sources()
        if sources:
            tool["sources"] = sources

        enable_image = (get_env("XAI_X_SEARCH_ENABLE_IMAGE_UNDERSTANDING") or "").strip().lower()
        if enable_image in {"true", "false"}:
            tool["enable_image_understanding"] = enable_image == "true"

        enable_video = (get_env("XAI_X_SEARCH_ENABLE_VIDEO_UNDERSTANDING") or "").strip().lower()
        if enable_video in {"true", "false"}:
            tool["enable_video_understanding"] = enable_video == "true"

        return tool

    def x_search(
        self,
        *,
        query: str,
        model_name: str = "grok",
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
        tool_choice: Optional[str] = None,
        sources: Optional[list[str]] = None,
        include_inline_citations: Optional[bool] = None,
        max_tool_calls: Optional[int] = None,
    ) -> ModelResponse:
        """Run xAI's native x_search tool and return the final text output.

        This is intended for PAL-native MCP tooling where the orchestrator wants to
        explicitly invoke x_search (instead of relying on prompt heuristics).
        """

        # Temporary overrides (tool-level)
        resolved_model = self._resolve_model_name(model_name)
        request: dict = {
            "model": resolved_model,
            "input": query,
            "tools": [self._build_x_search_tool()],
            "tool_choice": tool_choice or self._x_search_tool_choice(),
        }

        if sources is not None:
            request["tools"][0]["sources"] = [entry.strip().lower() for entry in sources if entry and entry.strip()]

        if system_prompt:
            request["instructions"] = system_prompt

        if include_inline_citations is None:
            include_inline = (get_env("XAI_INCLUDE_INLINE_CITATIONS", "false") or "").strip().lower() == "true"
        else:
            include_inline = include_inline_citations
        if include_inline:
            request["include"] = ["inline_citations"]

        if max_tool_calls is not None:
            request["max_tool_calls"] = int(max_tool_calls)
        else:
            max_tool_calls_raw = (get_env("XAI_X_SEARCH_MAX_TOOL_CALLS") or "").strip()
            if max_tool_calls_raw:
                try:
                    request["max_tool_calls"] = int(max_tool_calls_raw)
                except ValueError:
                    logger.warning("Ignoring invalid XAI_X_SEARCH_MAX_TOOL_CALLS: %s", max_tool_calls_raw)

        if max_output_tokens:
            request["max_output_tokens"] = max_output_tokens

        capabilities: Optional[ModelCapabilities] = None
        try:
            capabilities = self.get_capabilities(resolved_model)
        except Exception:
            capabilities = None

        effective_temperature = temperature
        if capabilities:
            adjusted = capabilities.get_effective_temperature(temperature)
            effective_temperature = adjusted

        if effective_temperature is not None:
            request["temperature"] = effective_temperature

        response = self.client.responses.create(**request)
        content = self._safe_extract_output_text(response)

        usage = None
        if hasattr(response, "usage") and getattr(response, "usage"):
            usage_obj = getattr(response, "usage")
            input_tokens = getattr(usage_obj, "input_tokens", None)
            output_tokens = getattr(usage_obj, "output_tokens", None)
            total_tokens = getattr(usage_obj, "total_tokens", None)

            if input_tokens is None:
                input_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
            if output_tokens is None:
                output_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
            if total_tokens is None:
                total_tokens = input_tokens + output_tokens

            usage = {
                "input_tokens": int(input_tokens) if input_tokens is not None else 0,
                "output_tokens": int(output_tokens) if output_tokens is not None else 0,
                "total_tokens": int(total_tokens) if total_tokens is not None else 0,
            }

        return ModelResponse(
            content=content,
            usage=usage,
            model_name=resolved_model,
            friendly_name=self.FRIENDLY_NAME,
            provider=self.get_provider_type(),
            metadata={
                "model": getattr(response, "model", resolved_model),
                "id": getattr(response, "id", ""),
                "created": getattr(response, "created_at", 0),
                "endpoint": "responses",
                "tool_choice": request.get("tool_choice"),
                "tools": request.get("tools"),
            },
        )

    def _generate_with_x_search(
        self,
        *,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str],
        temperature: float,
        max_output_tokens: Optional[int],
        capabilities: Optional[ModelCapabilities],
    ) -> ModelResponse:
        """Run xAI tool search via the Responses API and return the final text."""

        resolved_model = self._resolve_model_name(model_name)

        request: dict = {
            "model": resolved_model,
            "input": prompt,
            "tools": [self._build_x_search_tool()],
            "tool_choice": self._x_search_tool_choice(),
        }

        if system_prompt:
            request["instructions"] = system_prompt

        include_inline = (get_env("XAI_INCLUDE_INLINE_CITATIONS", "false") or "").strip().lower() == "true"
        if include_inline:
            request["include"] = ["inline_citations"]

        max_tool_calls_raw = (get_env("XAI_X_SEARCH_MAX_TOOL_CALLS") or "").strip()
        if max_tool_calls_raw:
            try:
                request["max_tool_calls"] = int(max_tool_calls_raw)
            except ValueError:
                logger.warning("Ignoring invalid XAI_X_SEARCH_MAX_TOOL_CALLS: %s", max_tool_calls_raw)

        if max_output_tokens:
            request["max_output_tokens"] = max_output_tokens

        effective_temperature = temperature
        if capabilities:
            adjusted = capabilities.get_effective_temperature(temperature)
            if adjusted is None:
                effective_temperature = None
            else:
                effective_temperature = adjusted

        if effective_temperature is not None:
            request["temperature"] = effective_temperature

        response = self.client.responses.create(**request)
        content = self._safe_extract_output_text(response)

        usage = None
        if hasattr(response, "usage") and getattr(response, "usage"):
            usage_obj = getattr(response, "usage")
            input_tokens = getattr(usage_obj, "input_tokens", None)
            output_tokens = getattr(usage_obj, "output_tokens", None)
            total_tokens = getattr(usage_obj, "total_tokens", None)

            if input_tokens is None:
                input_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
            if output_tokens is None:
                output_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
            if total_tokens is None:
                total_tokens = input_tokens + output_tokens

            usage = {
                "input_tokens": int(input_tokens) if input_tokens is not None else 0,
                "output_tokens": int(output_tokens) if output_tokens is not None else 0,
                "total_tokens": int(total_tokens) if total_tokens is not None else 0,
            }

        return ModelResponse(
            content=content,
            usage=usage,
            model_name=resolved_model,
            friendly_name=self.FRIENDLY_NAME,
            provider=self.get_provider_type(),
            metadata={
                "model": getattr(response, "model", resolved_model),
                "id": getattr(response, "id", ""),
                "created": getattr(response, "created_at", 0),
                "endpoint": "responses",
                "tool_choice": request.get("tool_choice"),
                "tools": request.get("tools"),
            },
        )

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using X.AI's OpenAI-style API, enabling native x_search when appropriate."""

        capabilities: Optional[ModelCapabilities] = None
        try:
            capabilities = self.get_capabilities(model_name)
        except Exception:
            # Capability lookup is best-effort; generation path will handle unknown models.
            capabilities = None

        if (
            self._x_search_enabled()
            and (capabilities is None or capabilities.supports_function_calling)
            and self._prompt_mentions_x_post(prompt)
        ):
            # Use xAI's agentic search tools via the Responses API.
            return self._generate_with_x_search(
                prompt=prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                capabilities=capabilities,
            )

        return super().generate_content(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            images=images,
            **kwargs,
        )

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> Optional[str]:
        """Get XAI's preferred model for a given category from allowed models.

        Args:
            category: The tool category requiring a model
            allowed_models: Pre-filtered list of models allowed by restrictions

        Returns:
            Preferred model name or None
        """
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        if category == ToolModelCategory.EXTENDED_REASONING:
            # Prefer Grok 4.1 Fast Reasoning for advanced tasks
            if self.PRIMARY_MODEL in allowed_models:
                return self.PRIMARY_MODEL
            if self.FALLBACK_MODEL in allowed_models:
                return self.FALLBACK_MODEL
            return allowed_models[0]

        elif category == ToolModelCategory.FAST_RESPONSE:
            # Prefer Grok 4.1 Fast Reasoning for speed as well (latest fast SKU).
            if self.PRIMARY_MODEL in allowed_models:
                return self.PRIMARY_MODEL
            if self.FALLBACK_MODEL in allowed_models:
                return self.FALLBACK_MODEL
            return allowed_models[0]

        else:  # BALANCED or default
            # Prefer Grok 4.1 Fast Reasoning for balanced use.
            if self.PRIMARY_MODEL in allowed_models:
                return self.PRIMARY_MODEL
            if self.FALLBACK_MODEL in allowed_models:
                return self.FALLBACK_MODEL
            return allowed_models[0]


# Load registry data at import time
XAIModelProvider._ensure_registry()
