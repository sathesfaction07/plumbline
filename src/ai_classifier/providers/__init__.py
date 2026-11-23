"""Provider registry.

`build` resolves a name to an instance, importing the cloud backends only when asked. The
offline provider is the default so that every command works with no credentials.
"""

from __future__ import annotations

from ai_classifier.providers.base import Provider
from ai_classifier.providers.offline import OfflineProvider

NAMES = ("offline", "bedrock", "vertex")


def build(name: str) -> Provider:
    """Resolve a provider name.

    Raises on an unknown name rather than falling back to offline. A typo silently
    downgrading a vision run to a text-only one would produce a full set of abstentions on
    image-only listings and look like a model problem.
    """
    if name == "offline":
        return OfflineProvider()
    if name == "bedrock":
        from ai_classifier.providers.bedrock import BedrockProvider  # noqa: PLC0415

        return BedrockProvider()
    if name == "vertex":
        from ai_classifier.providers.vertex import VertexProvider  # noqa: PLC0415

        return VertexProvider()
    raise ValueError(f"unknown provider {name!r}; choose one of {', '.join(NAMES)}")


__all__ = ["NAMES", "OfflineProvider", "Provider", "build"]
