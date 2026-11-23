"""Bedrock Claude, multimodal.

Imports boto3 inside the constructor rather than at module scope so that `import
ai_classifier` works without the extra installed. The alternative -- a top-level import
guarded by try/except -- means the failure surfaces as a confusing AttributeError at call
time instead of a clear message at construction.

Not exercised in CI: it needs credentials and a network. The parsing of what it returns IS
exercised, because that is where the interesting failures live -- `parse_response` is a
module-level function taking a plain dict for exactly that reason.
"""

from __future__ import annotations

import json

from ai_classifier.models import Classification, Modality, Product, Verdict
from ai_classifier.providers.base import Provider

PROMPT = """You are reviewing whether a flag on a product is correct.

Answer with JSON only: {"verdict": "true_positive"|"false_positive"|"not_enough_info",
"confidence": 0.0-1.0, "rationale": "one sentence"}

Rules that matter:
- Use not_enough_info when the evidence does not settle the question. An abstention is a
  useful answer; a guess dressed as a verdict is not.
- When you abstain, report a confidence at or below 0.5. Claiming to be certain that you
  are uncertain is incoherent and will be rejected.
- Judge only the evidence supplied. Do not infer from what a product like this usually is.
"""


def parse_response(payload: dict) -> tuple[Verdict, float, str]:
    """Pull a verdict out of a model response.

    Raises on anything it cannot read. A model that returns prose instead of JSON, or a
    verdict outside the enum, has not produced a classification -- and defaulting to
    NOT_ENOUGH_INFO here would silently convert a broken integration into a pile of
    plausible-looking abstentions, which is the failure mode hardest to notice.
    """
    try:
        verdict = Verdict(payload["verdict"])
    except (KeyError, ValueError) as error:
        raise ValueError(f"unusable verdict in response: {payload!r}") from error

    try:
        confidence = float(payload["confidence"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"unusable confidence in response: {payload!r}") from error

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence {confidence} is outside 0..1")

    return verdict, confidence, str(payload.get("rationale", ""))


class BedrockProvider(Provider):
    """Claude on Bedrock. Reads text and images."""

    name = "bedrock"

    def __init__(self, model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
                 region: str = "us-east-1") -> None:
        try:
            import boto3  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - depends on the install
            raise ImportError(
                "the bedrock provider needs boto3: pip install -e '.[bedrock]'"
            ) from error
        self.model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def supports(self, modality: Modality) -> bool:
        return modality is not Modality.NEITHER

    def classify(self, product: Product) -> Classification:  # pragma: no cover - network
        if not self.supports(product.modality):
            return self.unable(product, "the listing carries neither text nor images")

        content: list[dict] = [{"type": "text", "text": PROMPT}]
        if product.attributes:
            content.append({"type": "text", "text": product.text_blob()})
        for reference in product.images:
            content.append({"type": "text", "text": f"[image] {reference}"})

        response = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": content}],
            }),
        )
        body = json.loads(response["body"].read())
        verdict, confidence, rationale = parse_response(
            json.loads(body["content"][0]["text"])
        )
        return Classification(
            product_id=product.product_id,
            verdict=verdict,
            confidence=confidence,
            modality=product.modality,
            provider=self.name,
            rationale=rationale,
        )
