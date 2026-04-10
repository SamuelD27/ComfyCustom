#!/usr/bin/env python3
"""Caption cleanup script for identity LoRA training datasets.

Audits and cleans .txt caption files:
- Strips identity-defining descriptors (so they bind to the trigger word)
- Removes hedging language from AI-generated captions
- Normalizes formatting
- Validates trigger word placement
- Finds orphaned images/captions
"""

import argparse
import csv
import logging
import os
import re
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

log = logging.getLogger(__name__)


def load_blocklist(path: str) -> list[str]:
    """Load blocklist terms from a text file.

    Lines starting with # are comments. Blank lines are skipped.
    Raises FileNotFoundError if the file doesn't exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Blocklist file not found: {path}")
    terms = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                terms.append(stripped)
    return terms


def validate_trigger_word(caption: str, trigger: str) -> bool:
    """Check if the trigger word is present at the very start of the caption."""
    return caption.startswith(trigger)


def strip_identity_descriptors(caption: str, blocklist: list[str]) -> str:
    """Remove identity-defining descriptors from caption.

    Handles "with {term}" patterns and standalone terms.
    Case-insensitive matching. Cleans up leftover comma artifacts.
    """
    result = caption
    for term in blocklist:
        # Remove "with {term}" pattern (with optional leading comma/space)
        result = re.sub(
            r",?\s*\bwith\s+" + re.escape(term) + r"\b",
            "",
            result,
            flags=re.IGNORECASE,
        )
        # Remove standalone term (with optional surrounding commas/spaces)
        result = re.sub(
            r",?\s*\b" + re.escape(term) + r"\b\s*,?",
            ",",
            result,
            flags=re.IGNORECASE,
        )
    # Clean up comma artifacts
    result = re.sub(r",\s*,", ",", result)  # double commas
    result = re.sub(r"^,\s*", "", result)  # leading comma
    result = re.sub(r",\s*$", "", result)  # trailing comma
    result = re.sub(r"\s{2,}", " ", result)  # double spaces
    return result.strip()


def remove_hedging(caption: str) -> str:
    """Remove hedging language from AI-generated captions."""
    hedging_patterns = [
        r"\bappears\s+to\s+be\b",
        r"\bseems\s+to\s+be\b",
        r"\bseems\s+to\b",
        r"\bpossibly\b",
        r"\bprobably\b",
        r"\blikely\b",
        r"\bperhaps\b",
        r"\bmight\s+be\b",
        r"\bcould\s+be\b",
    ]
    result = caption
    for pattern in hedging_patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    # Clean up double spaces left by removal
    result = re.sub(r"\s{2,}", " ", result)
    return result.strip()


def normalize_formatting(caption: str) -> str:
    """Normalize caption formatting.

    - Fix comma spacing ("a ,b" -> "a, b")
    - Remove double commas
    - Remove double spaces
    - Strip trailing whitespace
    """
    result = caption
    # Fix space-before-comma: "a ,b" -> "a, b"
    result = re.sub(r"\s+,", ",", result)
    # Ensure space after comma
    result = re.sub(r",(?!\s)", ", ", result)
    # Remove double commas
    result = re.sub(r",\s*,", ",", result)
    # Remove double spaces
    result = re.sub(r"\s{2,}", " ", result)
    # Strip trailing whitespace
    result = result.rstrip()
    return result


def find_orphans(directory: str) -> tuple[list[str], list[str]]:
    """Find images without captions and captions without images.

    Returns (orphan_images, orphan_captions) as lists of filenames.
    """
    dir_path = Path(directory)
    image_stems = {}
    caption_stems = {}

    for f in dir_path.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            image_stems[f.stem] = f.name
        elif f.suffix.lower() == ".txt":
            caption_stems[f.stem] = f.name

    orphan_images = sorted(
        [name for stem, name in image_stems.items() if stem not in caption_stems]
    )
    orphan_captions = sorted(
        [name for stem, name in caption_stems.items() if stem not in image_stems]
    )

    return orphan_images, orphan_captions


def process_caption(
    caption: str,
    trigger: str,
    blocklist: list[str],
) -> str:
    """Full caption processing pipeline.

    1. Strip identity descriptors
    2. Remove hedging language
    3. Normalize formatting
    4. Ensure trigger word at start
    """
    result = caption
    result = strip_identity_descriptors(result, blocklist)
    result = remove_hedging(result)
    result = normalize_formatting(result)

    # Ensure trigger word at the start
    if not validate_trigger_word(result, trigger):
        # Prepend trigger word
        if result:
            result = f"{trigger}, {result}"
        else:
            result = trigger
    return result


def retrigger_caption(
    caption: str,
    old_trigger: str,
    new_trigger: str,
) -> str:
    """Replace old trigger word with new one, or prepend new if missing."""
    if caption.startswith(old_trigger):
        return new_trigger + caption[len(old_trigger):]
    # Old trigger not found at start, prepend new trigger
    if caption:
        return f"{new_trigger}, {caption}"
    return new_trigger


def main():
    parser = argparse.ArgumentParser(
        description="Clean up caption files for identity LoRA training datasets."
    )
    parser.add_argument(
        "directory",
        help="Path to dataset directory containing images and .txt caption files",
    )
    parser.add_argument(
        "--trigger",
        required=True,
        help="Trigger word that must appear at the start of each caption",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without modifying files",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a CSV report of changes",
    )
    parser.add_argument(
        "--retrigger",
        metavar="NEW",
        help="Replace current trigger word with NEW trigger word",
    )
    parser.add_argument(
        "--blocklist",
        default=os.path.join(os.path.dirname(__file__), "caption_blocklist.txt"),
        help="Path to blocklist file (default: scripts/caption_blocklist.txt)",
    )

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: directory not found: {args.directory}", file=sys.stderr)
        sys.exit(1)

    # Load blocklist
    blocklist = []
    if os.path.isfile(args.blocklist):
        blocklist = load_blocklist(args.blocklist)
        print(f"Loaded {len(blocklist)} blocklist terms from {args.blocklist}")
    else:
        print(f"No blocklist file found at {args.blocklist}, proceeding without blocklist")

    # Find orphans
    orphan_images, orphan_captions = find_orphans(args.directory)
    if orphan_images:
        print(f"\nOrphan images (no caption): {len(orphan_images)}")
        for name in orphan_images:
            print(f"  {name}")
    if orphan_captions:
        print(f"\nOrphan captions (no image): {len(orphan_captions)}")
        for name in orphan_captions:
            print(f"  {name}")

    # Process captions
    dir_path = Path(args.directory)
    caption_files = sorted(dir_path.glob("*.txt"))

    stats = {
        "total": 0,
        "modified": 0,
        "trigger_added": 0,
        "trigger_replaced": 0,
        "descriptors_stripped": 0,
        "hedging_removed": 0,
    }
    report_rows = []

    for caption_file in caption_files:
        original = caption_file.read_text().strip()
        stats["total"] += 1

        if args.retrigger:
            processed = retrigger_caption(original, old_trigger=args.trigger, new_trigger=args.retrigger)
            if original != processed:
                if original.startswith(args.trigger):
                    stats["trigger_replaced"] += 1
                else:
                    stats["trigger_added"] += 1
        else:
            processed = process_caption(original, trigger=args.trigger, blocklist=blocklist)

            # Track what changed
            if original != processed:
                if not validate_trigger_word(original, args.trigger):
                    stats["trigger_added"] += 1
                test_stripped = strip_identity_descriptors(original, blocklist)
                if test_stripped != original:
                    stats["descriptors_stripped"] += 1
                test_hedging = remove_hedging(original)
                if test_hedging != original:
                    stats["hedging_removed"] += 1

        if original != processed:
            stats["modified"] += 1
            if args.dry_run:
                print(f"\n--- {caption_file.name} ---")
                print(f"  OLD: {original}")
                print(f"  NEW: {processed}")
            else:
                caption_file.write_text(processed + "\n")

            if args.report:
                report_rows.append({
                    "file": caption_file.name,
                    "original": original,
                    "processed": processed,
                })

    # Write report
    if args.report and report_rows:
        report_path = os.path.join(args.directory, "caption_cleanup_report.csv")
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "original", "processed"])
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"\nReport written to {report_path}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"Total captions:       {stats['total']}")
    print(f"Modified:             {stats['modified']}")
    print(f"Trigger added:        {stats['trigger_added']}")
    if args.retrigger:
        print(f"Trigger replaced:     {stats['trigger_replaced']}")
    else:
        print(f"Descriptors stripped: {stats['descriptors_stripped']}")
        print(f"Hedging removed:      {stats['hedging_removed']}")
    print(f"Orphan images:        {len(orphan_images)}")
    print(f"Orphan captions:      {len(orphan_captions)}")

    if args.dry_run:
        print("\n(dry-run mode -- no files were modified)")


if __name__ == "__main__":
    main()
