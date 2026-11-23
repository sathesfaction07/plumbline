"""The provider interface.

A provider turns a product into a `Classification`. Three exist: an offline one that reads
text signals, and two cloud ones that send images plus text to a vision model.

The interface takes a whole `Product` rather than a prepared prompt, because deciding what
evidence to use IS the provider's job -- a text-only provider handed an image-only listing
has to say so, and that decision cannot be made by the caller without duplicating the
provider's knowledge of its own capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ai_classifier.models import Classification, Modality, Product, Verdict


class Provider(ABC):
    """Classifies products.

    Subclasses declare what they can read via `supports`, and the pipeline uses that to
    route rather than to filter: an unsupported product still gets a `Classification`, it
    just gets an honest NOT_ENOUGH_INFO one. Filtering would drop the product from the
    output entirely, so a reviewer would never learn it needed attention.
    """

    name: str = "provider"

    @abstractmethod
    def supports(self, modality: Modality) -> bool:
        """Whether this provider can read the evidence a product carries."""

    @abstractmethod
    def classify(self, product: Product) -> Classification:
        """Classify one product. Must not raise for unsupported input."""

    def unable(self, product: Product, why: str) -> Classification:
        """An honest abstention.

        Confidence is 0.0 rather than a low-but-nonzero value. There is no evidence here to
        be slightly confident about, and `Classification` refuses an abstention above 0.5
        anyway. Shared so that all three providers phrase "I can't" identically -- a
        reviewer filtering on rationale should not have to know which backend ran.
        """
        return Classification(
            product_id=product.product_id,
            verdict=Verdict.NOT_ENOUGH_INFO,
            confidence=0.0,
            modality=product.modality,
            provider=self.name,
            rationale=why,
        )
