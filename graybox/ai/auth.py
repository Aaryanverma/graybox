"""
Authentication manager for Gray Box.

Gray Box intentionally keeps authentication lightweight.

Responsibilities:
- API key authentication (no-op)
- Azure AD / Microsoft Entra authentication

All other provider-specific authentication and configuration
(e.g. AWS Bedrock, Google Vertex AI, Anthropic, OpenAI, etc.)
should be passed directly via LiteLLM kwargs.
"""

from __future__ import annotations

import os
from typing import Any


class AuthenticationManager:
    """Provides authentication kwargs for LiteLLM."""

    _AZURE_SCOPE = "https://cognitiveservices.azure.com/.default"

    _HANDLERS = {
        "api_key": "_api_key",
        "azure_ad": "_azure_ad",
    }

    _ALIASES = {
        "azure": "azure_ad",
        "entra": "azure_ad",
        "ms_entra": "azure_ad",
        "microsoft_entra": "azure_ad",
    }

    def __init__(self):
        self._cache: dict[str, Any] = {}

    def get_auth_kwargs(self, cfg) -> dict[str, Any]:
        auth = getattr(cfg, "auth", "api_key")
        auth = self._ALIASES.get(auth, auth)

        handler_name = self._HANDLERS.get(auth)

        if handler_name is None:
            raise ValueError(
                f"Unsupported authentication mode: {auth!r}. "
                f"Supported modes: {self._supported_modes()}"
            )

        return getattr(self, handler_name)(cfg)

    def _supported_modes(self) -> str:
        """Returns a human-readable list of supported authentication modes."""

        reverse_aliases: dict[str, list[str]] = {}

        for alias, canonical in self._ALIASES.items():
            reverse_aliases.setdefault(canonical, []).append(alias)

        supported = []

        for mode in self._HANDLERS:
            aliases = reverse_aliases.get(mode)

            if aliases:
                supported.append(f"{mode} (aliases: {', '.join(sorted(aliases))})")
            else:
                supported.append(mode)

        return ", ".join(supported)

    # ------------------------------------------------------------------ #
    # API Key
    # ------------------------------------------------------------------ #

    def _api_key(self, cfg) -> dict[str, Any]:
        """
        API keys are already passed separately by AIService.
        """
        return {}

    # ------------------------------------------------------------------ #
    # Azure AD / Microsoft Entra
    # ------------------------------------------------------------------ #

    def _azure_ad(self, cfg) -> dict[str, Any]:
        """
        Authentication priority:

        1. Configured Azure AD token
        2. AZURE_AD_TOKEN environment variable
        3. DefaultAzureCredential
        """

        token = getattr(cfg, "azure_ad_token", None) or os.environ.get("AZURE_AD_TOKEN")

        if token:
            return {
                "azure_ad_token": token,
            }

        provider = self._cache.get("azure_ad")

        if provider is None:
            provider = self._build_azure_provider()
            self._cache["azure_ad"] = provider

        return {
            "azure_ad_token_provider": provider,
        }

    def _build_azure_provider(self):
        """
        Lazily constructs a cached Azure AD token provider.

        DefaultAzureCredential automatically supports:

        - Environment credentials
        - Azure CLI (az login)
        - Azure Developer CLI
        - VS Code Azure extension
        - Managed Identity
        - Interactive browser login
        """

        try:
            from azure.identity import (
                DefaultAzureCredential,
                get_bearer_token_provider,
            )
        except ImportError as exc:
            raise ImportError(
                "Azure AD authentication requires the "
                "'azure-identity' package.\n\n"
                "Install it with:\n"
                "    pip install azure-identity"
            ) from exc

        credential = DefaultAzureCredential(
            exclude_interactive_browser_credential=False,
        )

        return get_bearer_token_provider(
            credential,
            self._AZURE_SCOPE,
        )
