"""Utility for constructing ComfyUI workflow JSON (UI format).

Builds the litegraph-compatible JSON that ComfyUI's frontend can load,
with nodes, links, groups, and proper cross-references.
"""

import json
from pathlib import Path


def make_workflow(nodes_def, links_def, groups_def=None):
    """Build a ComfyUI UI-format workflow dict.

    Args:
        nodes_def: list of dicts with keys:
            id (int), type (str), pos ([x, y]),
            widgets_values (list), inputs (list of (name, type_str) tuples),
            outputs (list of (name, type_str) tuples),
            optional: size ([w, h]), title (str), mode (int),
            color (str), bgcolor (str), properties (dict)
        links_def: list of tuples:
            (src_id, src_slot, dst_id, dst_slot, type_str)
        groups_def: optional list of dicts with:
            title (str), bounding ([x, y, w, h]), color (str)

    Returns:
        Complete workflow dict with version 0.4
    """
    # --- Build link objects with auto-incrementing IDs ---
    # Link format: [link_id, src_id, src_slot, dst_id, dst_slot, type_str]
    links = []
    dst_link_map = {}   # (dst_id, dst_slot) -> link_id
    src_link_map = {}   # (src_id, src_slot) -> [link_ids]

    for i, (src_id, src_slot, dst_id, dst_slot, type_str) in enumerate(links_def):
        link_id = i + 1
        links.append([link_id, src_id, src_slot, dst_id, dst_slot, type_str])
        dst_link_map[(dst_id, dst_slot)] = link_id
        src_link_map.setdefault((src_id, src_slot), []).append(link_id)

    # --- Build node objects ---
    nodes = []
    node_def_map = {n["id"]: n for n in nodes_def}

    for order, ndef in enumerate(nodes_def):
        nid = ndef["id"]

        # Build inputs with link references
        node_inputs = []
        for slot, (name, type_str) in enumerate(ndef.get("inputs", [])):
            link = dst_link_map.get((nid, slot))
            inp = {
                "name": name,
                "type": type_str,
                "link": link,
            }
            # Widget inputs (those that can also be set via widget) get a widget key
            if link is None and type_str in ("STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"):
                inp["widget"] = {"name": name}
            node_inputs.append(inp)

        # Build outputs with link arrays
        node_outputs = []
        for slot, (name, type_str) in enumerate(ndef.get("outputs", [])):
            out_links = src_link_map.get((nid, slot))
            node_outputs.append({
                "name": name,
                "type": type_str,
                "slot_index": slot,
                "links": out_links if out_links else None,
            })

        node = {
            "id": nid,
            "type": ndef["type"],
            "pos": ndef["pos"],
            "size": ndef.get("size", [315, 150]),
            "flags": {},
            "order": order,
            "mode": ndef.get("mode", 0),
            "inputs": node_inputs if node_inputs else [],
            "outputs": node_outputs if node_outputs else [],
            "properties": ndef.get("properties", {
                "Node name for S&R": ndef["type"],
            }),
            "widgets_values": ndef.get("widgets_values", []),
        }

        if "title" in ndef:
            node["title"] = ndef["title"]
        if "color" in ndef:
            node["color"] = ndef["color"]
        if "bgcolor" in ndef:
            node["bgcolor"] = ndef["bgcolor"]

        nodes.append(node)

    # --- Build groups ---
    groups = []
    if groups_def:
        for i, gdef in enumerate(groups_def):
            groups.append({
                "id": i + 1,
                "title": gdef["title"],
                "bounding": gdef["bounding"],
                "color": gdef.get("color", "#444"),
                "font_size": gdef.get("font_size", 24),
                "flags": {},
            })

    # --- Determine last IDs ---
    last_node_id = max(n["id"] for n in nodes_def) if nodes_def else 0
    last_link_id = len(links) if links else 0

    return {
        "last_node_id": last_node_id,
        "last_link_id": last_link_id,
        "nodes": nodes,
        "links": links,
        "groups": groups,
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def save_workflow(workflow, path):
    """Serialize workflow to JSON and save to disk.

    Args:
        workflow: workflow dict from make_workflow()
        path: str or Path destination
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
    print(f"Saved workflow to {p} ({p.stat().st_size:,} bytes)")
