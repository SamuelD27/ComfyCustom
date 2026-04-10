import os
import csv
import tempfile
import pytest
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from caption_cleanup import (
    load_blocklist,
    validate_trigger_word,
    strip_identity_descriptors,
    remove_hedging,
    normalize_formatting,
    find_orphans,
    process_caption,
    retrigger_caption,
)


class TestLoadBlocklist:
    def test_loads_terms(self, tmp_path):
        bl = tmp_path / "blocklist.txt"
        bl.write_text("brown eyes\nblonde hair\nslim build\n")
        result = load_blocklist(str(bl))
        assert result == ["brown eyes", "blonde hair", "slim build"]

    def test_skips_comments_and_blanks(self, tmp_path):
        bl = tmp_path / "blocklist.txt"
        bl.write_text("# This is a comment\n\nbrown eyes\n\n# Another comment\nblonde hair\n")
        result = load_blocklist(str(bl))
        assert result == ["brown eyes", "blonde hair"]

    def test_handles_empty_file(self, tmp_path):
        bl = tmp_path / "blocklist.txt"
        bl.write_text("")
        result = load_blocklist(str(bl))
        assert result == []

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_blocklist("/nonexistent/path/blocklist.txt")


class TestValidateTriggerWord:
    def test_present_at_start(self):
        assert validate_trigger_word("ohwx woman, a portrait in a park", "ohwx woman") is True

    def test_missing(self):
        assert validate_trigger_word("a portrait of a woman in a park", "ohwx woman") is False

    def test_present_but_not_at_start(self):
        assert validate_trigger_word("a portrait of ohwx woman in a park", "ohwx woman") is False

    def test_case_sensitive(self):
        assert validate_trigger_word("Ohwx woman, a portrait", "ohwx woman") is False


class TestStripIdentityDescriptors:
    def test_removes_single_term(self):
        result = strip_identity_descriptors(
            "ohwx woman, a woman with brown eyes wearing a red dress",
            ["brown eyes"]
        )
        assert "brown eyes" not in result

    def test_removes_multiple_terms(self):
        result = strip_identity_descriptors(
            "ohwx woman, a woman with brown eyes, blonde hair, wearing a dress",
            ["brown eyes", "blonde hair"]
        )
        assert "brown eyes" not in result
        assert "blonde hair" not in result

    def test_handles_with_prefix(self):
        result = strip_identity_descriptors(
            "ohwx woman, a woman with brown eyes wearing a dress",
            ["brown eyes"]
        )
        assert "with brown eyes" not in result

    def test_case_insensitive(self):
        result = strip_identity_descriptors(
            "ohwx woman, a woman with Brown Eyes wearing a dress",
            ["brown eyes"]
        )
        assert "Brown Eyes" not in result
        assert "brown eyes" not in result.lower()

    def test_no_double_commas_after_removal(self):
        result = strip_identity_descriptors(
            "ohwx woman, brown eyes, wearing a red dress",
            ["brown eyes"]
        )
        assert ",," not in result
        assert ", ," not in result


class TestRemoveHedging:
    def test_removes_appears_to_be(self):
        result = remove_hedging("The woman appears to be standing in a park")
        assert "appears to be" not in result
        assert "standing" in result

    def test_removes_seems_to(self):
        result = remove_hedging("She seems to be wearing a hat")
        assert "seems to" not in result

    def test_removes_possibly(self):
        result = remove_hedging("A possibly outdoor scene with trees")
        assert "possibly" not in result
        assert "outdoor" in result

    def test_removes_likely(self):
        result = remove_hedging("A likely professional photo of a woman")
        assert "likely" not in result
        assert "professional" in result

    def test_preserves_normal_text(self):
        text = "A woman wearing a red dress stands in a park"
        result = remove_hedging(text)
        assert result.strip() == text


class TestNormalizeFormatting:
    def test_removes_double_spaces(self):
        result = normalize_formatting("a  woman  in  a  park")
        assert "  " not in result

    def test_removes_double_commas(self):
        result = normalize_formatting("a woman,, wearing a dress")
        assert ",," not in result

    def test_strips_trailing_whitespace(self):
        result = normalize_formatting("a woman in a park   ")
        assert result == result.rstrip()

    def test_normalizes_comma_spacing(self):
        result = normalize_formatting("a woman ,wearing a dress")
        assert "a woman, wearing a dress" in result


class TestFindOrphans:
    def test_finds_images_without_captions(self, tmp_path):
        (tmp_path / "img1.png").touch()
        (tmp_path / "img2.jpg").touch()
        (tmp_path / "img1.txt").write_text("caption")
        orphan_images, orphan_captions = find_orphans(str(tmp_path))
        assert "img2.jpg" in orphan_images
        assert "img1.png" not in orphan_images

    def test_finds_captions_without_images(self, tmp_path):
        (tmp_path / "img1.png").touch()
        (tmp_path / "img1.txt").write_text("caption")
        (tmp_path / "img3.txt").write_text("caption")
        orphan_images, orphan_captions = find_orphans(str(tmp_path))
        assert "img3.txt" in orphan_captions
        assert "img1.txt" not in orphan_captions

    def test_handles_no_orphans(self, tmp_path):
        (tmp_path / "img1.png").touch()
        (tmp_path / "img1.txt").write_text("caption")
        (tmp_path / "img2.jpg").touch()
        (tmp_path / "img2.txt").write_text("caption")
        orphan_images, orphan_captions = find_orphans(str(tmp_path))
        assert orphan_images == []
        assert orphan_captions == []


class TestProcessCaption:
    def test_full_pipeline(self):
        caption = "a woman with brown eyes, appears to be  standing in a park"
        result = process_caption(
            caption,
            trigger="ohwx woman",
            blocklist=["brown eyes"],
        )
        # Should strip descriptors, remove hedging, normalize, and ensure trigger
        assert result.startswith("ohwx woman")
        assert "brown eyes" not in result
        assert "appears to be" not in result
        assert "  " not in result


class TestRetriggerCaption:
    def test_replaces_old_trigger_with_new(self):
        caption = "ohwx woman, a portrait in a park"
        result = retrigger_caption(caption, old_trigger="ohwx woman", new_trigger="sks woman")
        assert result.startswith("sks woman")
        assert "ohwx woman" not in result

    def test_adds_trigger_if_missing(self):
        caption = "a portrait of a woman in a park"
        result = retrigger_caption(caption, old_trigger="ohwx woman", new_trigger="sks woman")
        assert result.startswith("sks woman")
