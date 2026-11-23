"""Models, providers, loading, and batch checkpointing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_classifier.core.batch import Checkpoint, run_batch
from ai_classifier.io import load_labels, load_products, pair, product_from_row
from ai_classifier.models import (
    Classification,
    Label,
    Modality,
    Outcome,
    Product,
    Verdict,
)
from ai_classifier.providers import build
from ai_classifier.providers.base import Provider
from ai_classifier.providers.bedrock import parse_response
from tests.conftest import classification, product


class TestVerdict:
    def test_an_abstention_is_not_an_answer(self):
        assert Verdict.TRUE_POSITIVE.is_answer
        assert Verdict.FALSE_POSITIVE.is_answer
        assert not Verdict.NOT_ENOUGH_INFO.is_answer


class TestModality:
    @pytest.mark.parametrize(
        ("text", "images", "expected"),
        [
            ("has words", 2, Modality.BOTH),
            ("has words", 0, Modality.TEXT_ONLY),
            ("", 2, Modality.IMAGE_ONLY),
            ("", 0, Modality.NEITHER),
        ],
    )
    def test_modality_follows_the_evidence(self, text, images, expected):
        assert product(text=text, images=images).modality is expected

    def test_a_blank_attribute_is_not_text(self):
        """A feed emitting description="" must not present as TEXT_ONLY."""
        item = Product(product_id="P-1", attributes={"description": "   "}, images=["a.jpg"])
        assert item.attributes == {}
        assert item.modality is Modality.IMAGE_ONLY

    def test_has_text_and_has_image_agree_with_the_modality(self):
        assert Modality.BOTH.has_text and Modality.BOTH.has_image
        assert Modality.IMAGE_ONLY.has_image and not Modality.IMAGE_ONLY.has_text
        assert not Modality.NEITHER.has_text and not Modality.NEITHER.has_image

    def test_the_text_blob_is_sorted_so_the_prompt_is_stable(self):
        one = Product(product_id="P", attributes={"b": "2", "a": "1"})
        two = Product(product_id="P", attributes={"a": "1", "b": "2"})
        assert one.text_blob() == two.text_blob() == "a: 1\nb: 2"


class TestClassificationRefusesIncoherentOutput:
    def test_a_confident_abstention_is_rejected(self):
        """"I cannot tell, and I am 95% sure" is not a coherent answer."""
        with pytest.raises(ValidationError, match="cannot be held confidently"):
            Classification(
                product_id="P-1",
                verdict=Verdict.NOT_ENOUGH_INFO,
                confidence=0.95,
                modality=Modality.BOTH,
                provider="test",
            )

    def test_a_low_confidence_abstention_is_fine(self):
        item = Classification(
            product_id="P-1", verdict=Verdict.NOT_ENOUGH_INFO, confidence=0.4,
            modality=Modality.BOTH, provider="test",
        )
        assert item.verdict is Verdict.NOT_ENOUGH_INFO

    def test_confidence_outside_zero_to_one_is_rejected(self):
        with pytest.raises(ValidationError):
            classification(confidence=1.4)


class TestLabelMustBeGroundTruth:
    def test_an_undecided_label_is_refused(self):
        with pytest.raises(ValidationError, match="not ground truth"):
            Label(product_id="P-1", verdict=Verdict.NOT_ENOUGH_INFO)

    def test_a_decided_label_is_accepted(self):
        assert Label(product_id="P-1", verdict=Verdict.FALSE_POSITIVE).source == "human"


class TestOutcome:
    def test_correct_is_three_valued(self):
        right = Outcome(
            classification=classification(verdict=Verdict.TRUE_POSITIVE),
            label=Label(product_id="P-1", verdict=Verdict.TRUE_POSITIVE),
        )
        wrong = Outcome(
            classification=classification(verdict=Verdict.TRUE_POSITIVE),
            label=Label(product_id="P-1", verdict=Verdict.FALSE_POSITIVE),
        )
        declined = Outcome(
            classification=classification(verdict=Verdict.NOT_ENOUGH_INFO, confidence=0.0),
            label=Label(product_id="P-1", verdict=Verdict.TRUE_POSITIVE),
        )
        assert right.correct is True
        assert wrong.correct is False
        assert declined.correct is None, "an abstention is not an error"
        assert declined.answered is False

    def test_pairing_two_different_products_is_refused(self):
        with pytest.raises(ValidationError, match="pairs P-1 with a label for P-2"):
            Outcome(
                classification=classification(product_id="P-1"),
                label=Label(product_id="P-2", verdict=Verdict.TRUE_POSITIVE),
            )


class TestOfflineProvider:
    def test_it_abstains_on_an_image_only_listing(self):
        result = build("offline").classify(product(text="", images=3))
        assert result.verdict is Verdict.NOT_ENOUGH_INFO
        assert result.confidence == 0.0
        assert "cannot see images" in result.rationale

    def test_it_abstains_when_no_known_signal_appears(self):
        result = build("offline").classify(product(text="a perfectly ordinary object"))
        assert result.verdict is Verdict.NOT_ENOUGH_INFO
        assert "no known signal" in result.rationale

    def test_it_abstains_when_signals_cancel_exactly(self):
        """The ambiguous-product case: picking a side would need an arbitrary confidence."""
        result = build("offline").classify(product(text="contains lithium and non flammable"))
        assert result.verdict is Verdict.NOT_ENOUGH_INFO
        assert "cancel" in result.rationale

    def test_supporting_signals_produce_a_true_positive(self):
        result = build("offline").classify(product(text="contains lithium battery pack"))
        assert result.verdict is Verdict.TRUE_POSITIVE
        assert result.confidence > 0.9

    def test_contradicting_signals_produce_a_false_positive(self):
        result = build("offline").classify(product(text="battery free, no batteries at all"))
        assert result.verdict is Verdict.FALSE_POSITIVE

    def test_a_mixed_signal_lowers_the_confidence(self):
        clear = build("offline").classify(product("A", text="contains lithium"))
        mixed = build("offline").classify(product("B", text="contains lithium and plush"))
        assert mixed.confidence < clear.confidence

    def test_a_negation_does_not_fire_the_term_it_negates(self):
        """"flammable" is a substring of "non flammable".

        Scanning the two tables independently made a pure contradiction score 3 supporting
        AND 3 contradicting, cancelling to exactly zero -- so the provider abstained on
        unambiguous evidence. An abstention reads as caution, so nothing in the output
        suggested a matching defect.
        """
        result = build("offline").classify(product(text="this item is non flammable"))
        assert result.verdict is Verdict.FALSE_POSITIVE
        assert "non flammable" in result.rationale
        assert "margin -3" in result.rationale

    def test_masking_does_not_suppress_a_genuine_longer_match(self):
        """The mask must only consume the span it matched, not the rest of the text."""
        result = build("offline").classify(product(text="contains flammable aerosol"))
        assert result.verdict is Verdict.TRUE_POSITIVE
        assert "margin +6" in result.rationale, "both terms must still count"

    def test_it_is_deterministic(self):
        item = product(text="contains aerosol and magnet")
        first = build("offline").classify(item)
        second = build("offline").classify(item)
        assert first.model_dump() == second.model_dump()


class TestProviderRegistry:
    def test_an_unknown_name_raises_rather_than_falling_back(self):
        """A typo silently downgrading to offline would look like a model problem."""
        with pytest.raises(ValueError, match="unknown provider"):
            build("bedroc")

    def test_offline_is_available_with_no_extras(self):
        assert build("offline").name == "offline"


class TestBedrockResponseParsing:
    """The network call is not tested; the parsing of what comes back is."""

    def test_a_well_formed_response_parses(self):
        verdict, confidence, why = parse_response(
            {"verdict": "true_positive", "confidence": 0.88, "rationale": "clear"}
        )
        assert verdict is Verdict.TRUE_POSITIVE
        assert confidence == 0.88
        assert why == "clear"

    @pytest.mark.parametrize(
        "payload",
        [
            {"confidence": 0.9},
            {"verdict": "maybe", "confidence": 0.9},
            {"verdict": "true_positive"},
            {"verdict": "true_positive", "confidence": "high"},
            {"verdict": "true_positive", "confidence": 1.5},
        ],
    )
    def test_an_unusable_response_raises_rather_than_defaulting(self, payload):
        """Defaulting to not_enough_info would turn a broken integration into abstentions."""
        with pytest.raises(ValueError):
            parse_response(payload)


class TestLoading:
    def test_a_row_with_no_id_is_collected_not_fatal(self, tmp_path):
        path = tmp_path / "p.csv"
        path.write_text("product_id,title\n,no id here\nP-2,fine\n", encoding="utf-8")
        report = load_products(path)
        assert len(report.products) == 1
        assert len(report.problems) == 1
        assert "no product id" in report.problems[0]

    def test_a_byte_order_mark_does_not_hide_the_id_column(self, tmp_path):
        path = tmp_path / "p.csv"
        path.write_bytes("product_id,title\nP-1,thing\n".encode("utf-8-sig"))
        assert load_products(path)[0] if False else len(load_products(path).products) == 1

    def test_images_split_on_comma_or_pipe(self):
        item = product_from_row({"product_id": "P-1", "images": "a.jpg|b.jpg,c.jpg"})
        assert item.images == ["a.jpg", "b.jpg", "c.jpg"]

    def test_an_undecided_label_is_reported_not_dropped_silently(self, tmp_path):
        path = tmp_path / "l.csv"
        path.write_text(
            "product_id,verdict\nP-1,true_positive\nP-2,not_enough_info\n", encoding="utf-8"
        )
        report = load_labels(path)
        assert len(report.labels) == 1
        assert report.undecided_labels == ["P-2"]

    def test_an_unrecognised_verdict_is_a_problem(self, tmp_path):
        path = tmp_path / "l.csv"
        path.write_text("product_id,verdict\nP-1,probably\n", encoding="utf-8")
        report = load_labels(path)
        assert not report.labels
        assert "is not a verdict" in report.problems[0]

    def test_jsonl_products_load(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text(
            '{"product_id":"P-1","description":"contains lithium"}\n'
            '{"product_id":"P-2","description":"plush"}\n',
            encoding="utf-8",
        )
        assert len(load_products(path).products) == 2

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_products(tmp_path / "nope.csv")


class TestPairing:
    def test_both_kinds_of_miss_are_reported(self):
        classifications = [classification("P-1"), classification("P-2")]
        labels = [
            Label(product_id="P-2", verdict=Verdict.TRUE_POSITIVE),
            Label(product_id="P-3", verdict=Verdict.TRUE_POSITIVE),
        ]
        report = pair(classifications, labels)
        assert [item.product_id for item in report.outcomes] == ["P-2"]
        assert report.unlabelled == ["P-1"]
        assert report.unclassified == ["P-3"]

    def test_duplicate_classifications_raise(self):
        with pytest.raises(ValueError, match="duplicate product ids"):
            pair([classification("P-1"), classification("P-1")], [])

    def test_duplicate_labels_raise(self):
        labels = [
            Label(product_id="P-1", verdict=Verdict.TRUE_POSITIVE),
            Label(product_id="P-1", verdict=Verdict.FALSE_POSITIVE),
        ]
        with pytest.raises(ValueError, match="duplicate product ids"):
            pair([], labels)


class TestCheckpointResume:
    def test_a_resumed_run_does_not_repeat_work(self, tmp_path):
        items = [product(f"P-{n}", text="contains lithium") for n in range(5)]
        path = tmp_path / "ck.jsonl"

        first = run_batch(items[:3], build("offline"), checkpoint_path=path)
        assert first.newly_classified == 3 and first.from_checkpoint == 0

        second = run_batch(items, build("offline"), checkpoint_path=path)
        assert second.from_checkpoint == 3
        assert second.newly_classified == 2
        assert second.total == 5

    def test_a_truncated_final_line_is_reported_and_skipped(self, tmp_path):
        path = tmp_path / "ck.jsonl"
        items = [product(f"P-{n}", text="contains lithium") for n in range(3)]
        run_batch(items, build("offline"), checkpoint_path=path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"product_id": "P-9", "verdi')

        result = run_batch(items, build("offline"), checkpoint_path=path)
        assert result.truncated_checkpoint_lines == [4]
        assert result.from_checkpoint == 3

    def test_resuming_under_a_different_provider_is_refused(self, tmp_path):
        path = tmp_path / "ck.jsonl"
        checkpoint = Checkpoint(path)
        checkpoint.append(classification("P-1", provider="vertex"))

        class Fake(Provider):
            name = "offline"

            def supports(self, modality):
                return True

            def classify(self, item):
                return classification(item.product_id)

        with pytest.raises(ValueError, match="was written by 'vertex'"):
            run_batch([product("P-2")], Fake(), checkpoint_path=path)

    def test_a_provider_failure_does_not_end_the_run(self, tmp_path):
        class Flaky(Provider):
            name = "flaky"

            def supports(self, modality):
                return True

            def classify(self, item):
                if item.product_id == "P-2":
                    raise RuntimeError("transient")
                return classification(item.product_id, provider="flaky")

        items = [product(f"P-{n}") for n in range(1, 4)]
        result = run_batch(items, Flaky())
        assert result.total == 2
        assert "P-2" in result.failed
        assert "RuntimeError: transient" == result.failed["P-2"]

    def test_a_failure_is_not_checkpointed_so_it_is_retried(self, tmp_path):
        path = tmp_path / "ck.jsonl"
        attempts: list[str] = []

        class Once(Provider):
            name = "once"

            def supports(self, modality):
                return True

            def classify(self, item):
                attempts.append(item.product_id)
                if len(attempts) == 1:
                    raise RuntimeError("first attempt fails")
                return classification(item.product_id, provider="once")

        item = [product("P-1")]
        assert run_batch(item, Once(), checkpoint_path=path).failed
        assert run_batch(item, Once(), checkpoint_path=path).total == 1
        assert attempts == ["P-1", "P-1"], "the failure must not have been recorded"

    def test_no_checkpoint_path_means_no_file_is_written(self, tmp_path):
        result = run_batch([product("P-1", text="contains lithium")], build("offline"))
        assert result.total == 1
        assert list(tmp_path.iterdir()) == []
