"""Vertex Gemini, multimodal.

Same shape as the Bedrock provider and for the same reasons: deferred import, response
parsing lifted out into a testable module-level function, no CI coverage of the call
itself.

Kept as a separate class rather than a `provider="gemini"` flag on one class. The two
differ in request construction, response shape, and auth, and a single class handling both
would be a body of branching that only ever runs one branch per deployment.
"""

from __future__ import annotations

from ai_classifier.models import Classification, Modality, Product
from ai_classifier.providers.base import Provider
from ai_classifier.providers.bedrock import PROMPT, parse_response


class VertexProvider(Provider):
    """Gemini on Vertex AI. Reads text and images."""

    name = "vertex"

    def __init__(self, model_name: str = "gemini-1.5-pro", project: str | None = None,
                 location: str = "us-central1") -> None:
        try:
            import vertexai  # noqa: PLC0415
            from vertexai.generative_models import GenerativeModel  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - depends on the install
            raise ImportError(
                "the vertex provider needs google-cloud-aiplatform: "
                "pip install -e '.[vertex]'"
            ) from error
        vertexai.init(project=project, location=location)
        self._model = GenerativeModel(model_name)

    def supports(self, modality: Modality) -> bool:
        return modality is not Modality.NEITHER

    def classify(self, product: Product) -> Classification:  # pragma: no cover - network
        if not self.supports(product.modality):
            return self.unable(product, "the listing carries neither text nor images")

        import json  # noqa: PLC0415

        parts = [PROMPT]
        if product.attributes:
            parts.append(product.text_blob())
        parts.extend(f"[image] {reference}" for reference in product.images)

        response = self._model.generate_content(parts)
        verdict, confidence, rationale = parse_response(json.loads(response.text))
        return Classification(
            product_id=product.product_id,
            verdict=verdict,
            confidence=confidence,
            modality=product.modality,
            provider=self.name,
            rationale=rationale,
        )
