#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _map_field(field_name: str, spec: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    spec = spec or {}
    t = spec.get("type")
    desc = spec.get("description")
    min_len = spec.get("min_len")
    max_len = spec.get("max_len")
    optional_flag = spec.get("optional")
    is_required = not optional_flag if optional_flag is not None else True

    def add_len_constraints(obj: dict[str, Any]) -> dict[str, Any]:
        if max_len is not None:
            # Interpret "words" limits as maxLength as a simple approximation
            obj["maxLength"] = max_len
        if min_len is not None and obj.get("type") == "string":
            obj["minLength"] = min_len
        return obj

    if t in ("str", "string"):
        prop = {"type": "string"}
        if desc:
            prop["description"] = desc
        return add_len_constraints(prop), is_required

    if t in ("int", "integer"):
        prop = {"type": "integer"}
        if desc:
            prop["description"] = desc
        return prop, is_required

    if t == "str_list":
        items: dict[str, Any] = {"type": "string"}
        # If exactly one char per element is desired, YAML example uses description; add a pattern helper if max_len==1
        if max_len == 1:
            items["pattern"] = "^.{1}$"
        prop = {"type": "array", "items": items}
        if desc:
            prop["description"] = desc
        if min_len is not None:
            prop["minItems"] = min_len
        if max_len is not None:
            prop["maxItems"] = max_len
        return prop, is_required

    if t == "item_list":
        prop = {
            "type": "array",
            "items": {"$ref": "#/$defs/Item"},
        }
        if desc:
            prop["description"] = desc
        if min_len is not None:
            prop["minItems"] = min_len
        if max_len is not None:
            prop["maxItems"] = max_len
        return prop, is_required

    if t == "content_list":
        prop = {
            "type": "array",
            "items": {"$ref": "#/$defs/BaseContent"},
        }
        if desc:
            prop["description"] = desc
        if min_len is not None:
            prop["minItems"] = min_len
        if max_len is not None:
            prop["maxItems"] = max_len
        return prop, is_required

    if t == "content":
        prop = {"$ref": "#/$defs/BaseContent"}
        if desc:
            prop["description"] = desc
        return prop, is_required

    if t == "image":
        prop = {"$ref": "#/$defs/BasicImage"}
        if desc:
            prop["description"] = desc
        return prop, is_required

    # Fallback: treat unknown as free-form string
    prop = {"type": "string"}
    if desc:
        prop["description"] = desc
    return prop, is_required


def _page_to_schema(page_key: str, page_spec: dict[str, Any], allowed_types: set[str]) -> dict[str, Any]:
    title_text = page_spec.get("description") or page_key

    properties: dict[str, Any] = {}
    required = ["type"]

    # fixed type const with validation against allowed types
    page_type = page_spec.get("type")
    if not page_type:
        raise ValueError(f"Page '{page_key}' must specify a 'type' and it must be one of: {sorted(allowed_types)}")
    if page_type not in allowed_types:
        raise ValueError(
            f"Page '{page_key}' has type '{page_type}' not present in type_map. Allowed types: {sorted(allowed_types)}"
        )
    properties["type"] = {"const": str(page_type)}

    # other fields
    for field, spec in page_spec.items():
        if field in {"description", "type"}:
            continue
        properties[field], field_required = _map_field(field, spec)
        if field_required:
            required.append(field)

    return {
        "type": "object",
        "title": title_text,
        "properties": properties,
        "required": sorted(set(required)),
        "additionalProperties": False,
    }


def build_schema(yaml_root: dict[str, Any]) -> dict[str, Any]:
    # build allowed types from type_map in YAML (list of single-key mappings)
    allowed_types: set[str] = set()
    tm = yaml_root.get("type_map")
    if isinstance(tm, list):
        for item in tm:
            if isinstance(item, dict):
                for t in item.keys():
                    allowed_types.add(str(t))
    # Fallback: infer from page specs if type_map missing
    if not allowed_types:
        for key, value in yaml_root.items():
            if isinstance(value, dict) and key != "type_map":
                t = value.get("type")
                if t:
                    allowed_types.add(str(t))

    one_of = []
    # iterate keys excluding type_map
    for key, value in yaml_root.items():
        if key == "type_map":
            continue
        if not isinstance(value, dict):
            continue
        one_of.append(_page_to_schema(key, value, allowed_types))

    schema: dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Slide Deck Structure",
        "description": "Schema to represent a structured set of slides for presentations.",
        "type": "object",
        "properties": {
            "slides": {
                "type": "array",
                "items": {"oneOf": one_of},
            }
        },
        "required": ["slides"],
        "additionalProperties": False,
        "$defs": {
            # Minimal defs aligned with template.schema.json
            "BaseContent": {
                "type": "object",
                "discriminator": {
                    "propertyName": "content_type",
                    "mapping": {
                        "text": "#/$defs/TextContent",
                        "image": "#/$defs/ImageContent",
                        "table": "#/$defs/TableContent",
                    },
                },
                "properties": {
                    "content_type": {
                        "type": "string",
                        "enum": ["text", "image", "table"],
                    }
                },
                "required": ["content_type"],
            },
            "BasicImage": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "format": "uri"},
                },
                "required": ["image_url"],
                "additionalProperties": False,
            },
            "Paragraph": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "bullet": {"type": "boolean", "default": False},
                    "level": {"type": "integer", "minimum": 0},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            "Item": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "maxLength": 4},
                    "content": {"type": "string", "maxLength": 10},
                },
                "required": ["title", "content"],
                "additionalProperties": False,
            },
            "TextContent": {
                "allOf": [
                    {"$ref": "#/$defs/BaseContent"},
                    {
                        "type": "object",
                        "properties": {
                            "paragraph": {
                                "oneOf": [
                                    {"type": "array", "items": {"$ref": "#/$defs/Paragraph"}, "minItems": 1},
                                    {"type": "string"},
                                ]
                            }
                        },
                        "required": ["paragraph"],
                    },
                ]
            },
            "ImageContent": {
                "allOf": [
                    {"$ref": "#/$defs/BaseContent"},
                    {
                        "type": "object",
                        "properties": {
                            "image_url": {"type": "string", "format": "uri"},
                            "caption": {"type": "string", "maxLength": 20},
                        },
                        "required": ["image_url"],
                    },
                ],
            },
            "TableContent": {
                "allOf": [
                    {"$ref": "#/$defs/BaseContent"},
                    {
                        "type": "object",
                        "properties": {
                            "header": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                            "rows": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                "minItems": 1,
                                "maxItems": 7,
                            },
                            "caption": {"type": "string", "maxLength": 20},
                            "n_rows": {"type": "integer", "minimum": 1, "maximum": 7},
                            "n_cols": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "required": ["header", "rows", "n_rows", "n_cols"],
                    },
                ]
            },
        },
    }
    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSON Schema from PPT YAML template")
    parser.add_argument("input", type=Path, help="Path to YAML template (e.g., yaml_example.yaml)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to write schema JSON (default: schema/auto.schema.json next to input)",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        yaml_root: dict[str, Any] = yaml.safe_load(f)

    schema = build_schema(yaml_root)

    out = args.output
    if out is None:
        base_dir = args.input.parent
        out = base_dir / "schema" / "auto.schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=4)

    print(f"Schema written to: {out}")


if __name__ == "__main__":
    main()
