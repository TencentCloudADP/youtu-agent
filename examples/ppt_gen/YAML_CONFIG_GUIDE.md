# PPT YAML Configuration Guide

> [中文YAML配置指南](./YAML_CONFIG_GUIDE.zh.md)

This document explains how to describe PPT templates with YAML so that the `ppt_gen` example can generate consistent slides across different topics and layouts.

## YAML file layout

A template YAML file (see [`yaml_example.yaml`](./yaml_example.yaml)) is composed of two sections:

1. **`type_map`** – an ordered list that maps every slide `type` to the zero-based slide index in the reference PPT template. The renderer uses these indices to duplicate the correct base slide before populating fields.
2. **Page definition blocks** – each `<name>_page` entry defines the schema for a slide type. Every block must include:
   - `description`: free-form docstring used for both human guidance and JSON schema.
   - `type`: the logical slide type (must exist in `type_map`).
   - One or more **field specifications** that describe the expected payload for each placeholder.

### Field specification keys

Each field under a page block can use the following attributes:

| Key          | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `type`       | Semantic data type (see the supported list below).                          |
| `description`| Optional hint given to the LLM and included in the JSON schema.            |
| `min_len`    | Lower bound for strings or list lengths (converted to `minLength`/`minItems`). |
| `max_len`    | Upper bound for strings or list lengths (`maxLength`/`maxItems`).          |
| `optional`   | Set to `true` to mark a field as optional; defaults to required.           |

### Supported field types

| Field type        | Meaning / Rendering behavior |
|-------------------|------------------------------|
| `str` / `int`     | Plain text or integer written directly into the named shape. `int` is cast to string before rendering. |
| `content`         | Rich block rendered through `handle_content`, accepting the structures listed below (`TextContent`, `ImageContent`, `TableContent`). |
| `content_list`    | Ordered array of `BaseContent` (same payload as `content`); entries are mapped onto `<field_name>1`, `<field_name>2`, etc. |
| `item_list`       | Array of `Item` objects (title + content) rendered with `handle_item`. |
| `str_list`        | List of short labels (e.g., SWOT letters) mapped to `label1`, `label2`, ... shapes. |
| `image`           | `BasicImage` with an `image_url`, placed into an image placeholder. |

### Payload structure reference

These field types serialize directly into the pydantic models defined in [`ppt_template_model.py`](./ppt_template_model.py):

1. **`content` / `content_list`** → `BaseContent` union (`content_type` chooses the concrete model):
   - `TextContent`: `{"content_type": "text", "paragraph": [Paragraph]|str}` where `Paragraph` objects carry `{text, bullet, level}`.
   - `ImageContent`: `{"content_type": "image", "image_url": "https://...", "caption": "optional"}`.
   - `TableContent`: `{"content_type": "table", "header": [str], "rows": [[str]], "n_rows": int, "n_cols": int, "caption": "optional"}`.

2. **`item_list`** → list of `Item` objects: `{"title": "2-4 words", "content": "≤10 words"}`. Rendering pairs each entry with shapes named `item_title{n}` / `item_content{n}`.

3. **`image`** → single `BasicImage`: `{"image_url": "https://..."}`. The renderer downloads and inserts it into the placeholder, deleting the original shape afterward.

4. **`str_list`** → simple string array. The YAML `max_len`/`min_len` constraints are propagated into the JSON schema; rendering targets shapes named `label1`, `label2`, ... in order.

5. **`content_list`** (additional detail): each entry is treated like a standalone `content` field. Name your shapes `<field_name>1`, `<field_name>2`, etc., to match the order.

> ⚠️ **Template labeling:** Ensure the shapes on the PPT slide use the same names as the YAML fields so `fill_template_with_yaml_config` can find and replace the correct placeholders.

## How the YAML becomes a JSON Schema

When you run [`main.py`](./main.py), it loads the YAML file (`--yaml_path`) and immediately calls `build_schema` to inject the generated JSON Schema into the agent instructions @examples/ppt_gen/main.py#39-84. No manual conversion is required.

Behind the scenes, `gen_schema.build_schema` performs the following steps:

1. `type_map` establishes the allowed slide `type` values.
2. Each page block becomes a `oneOf` entry whose `properties` mirror the field specs.
3. Length constraints (`min_len`, `max_len`) are translated to JSON Schema keywords.
4. Complex field types (`content`, `item_list`, `image`, etc.) are mapped to `$defs` references so validation matches the renderer’s expectations.

> Optional: You can still run `gen_schema.py` manually if you want to inspect the emitted schema file, but the main workflow handles it automatically.

## Adding a custom page type (example)

Suppose you need a new slide that highlights three metrics with icons.

1. **Prepare the PPT template:**
   - Duplicate an existing slide or design a new one.
   - Name the shapes in the selection pane to match the future YAML fields (`title`, `metric1_title`, etc.).

2. **Extend `type_map`:** add a new entry with the next free template index.

```yaml
type_map:
  - content: 0
  - title: 1
  - insight_grid: 13   # new slide lives at index 13 in the template
```

3. **Define the page block:**

```yaml
insight_grid_page:
  description: highlights three quantitative insights side-by-side
  type: insight_grid
  title:
    type: str
    description: concise headline within 6 words
    max_len: 6
  items:
    type: item_list
    description: each entry holds metric name + short explanation
    min_len: 3
    max_len: 3
  image:
    type: image
    optional: true
```

4. **Regenerate the schema:** run the `gen_schema.py` command so the agent knows about `insight_grid`.

5. **Run the generator:** invoke `python main.py ... --yaml_path <your_yaml>`; the agent will emit slides with `type: insight_grid`, and `fill_template_with_yaml_config` will duplicate slide index 13 and populate the named fields.

By keeping the YAML, JSON schema, and PPT template shapes in sync, you can introduce new layouts without modifying Python code.
