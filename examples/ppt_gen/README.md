# PowerPoint Template Example

> [中文README](README.zh.md)

## Prerequisites

This example requires additional dependencies for image processing, document handling, and visualization.

**Install dependencies:**

```bash
# Navigate to project root
cd youtu-agent

# Install PPT generation dependencies
uv sync --extra ppt-gen
```

## Quick Start

Download [template files](https://cdn.jsdelivr.net/gh/TencentCloudADP-DevRel/picgo-images@main/assets/templates.zip) to `examples/ppt_gen/template` directory.

Navigate to the example directory, prepare the reference resource (plain text / markdown / html webpage) for PPT generation. For example, download the [Nobel Prize webpage](https://www.nobelprize.org/prizes/physics/2025/popular-information/).

```bash
# Navigate to the example directory
cd examples/ppt_gen

# Download sample webpage
wget https://www.nobelprize.org/prizes/physics/2025/popular-information/ -O webpage.html
```

Run the PPT generation script.

```python
python main.py \
  --file webpage.html \
  --template_path templates/0.pptx \
  --yaml_path yaml_example.yaml \
  --pages 10 \
  --disable_tooluse \
  --extra_prompt "Language should be English."
```

The script will produce a `json` file and a `pptx` file if everything is OK. Pass `--yaml_path` to switch between different template definitions.

## YAML-driven template configuration

For more details, see [YAML-driven template configuration](./YAML_CONFIG_GUIDE.md).

The template workflow is powered by a YAML config (see `yaml_example.yaml`). The config plays two roles:

1. **type_map** — maps slide `type` values (e.g., `title`, `items_page_4`) to the slide index inside the reference template. These indices tell the renderer which base slide to duplicate.
2. **Page definitions** — each `<name>_page` block describes all fields that can appear on that slide. Every field specifies `type`, optional length constraints, and additional hints that are injected into the LLM’s JSON schema via `gen_schema.build_schema`.

Supported field types include `str`, `int`, `content` (rich text/image/table payload), `content_list`, `item_list`, `str_list`, and `image`. When the agent finishes generating content, `fill_template_with_yaml_config` reads the YAML config, looks up the target slide via `type_map`, duplicates the correct template page, and renders each field based on the declared type mappings. To customize a template:

1. Duplicate `yaml_example.yaml`, adjust `type_map` to match the slide order in your PPT template.
2. Update or add page blocks so that every slide type you want the agent to produce has the correct field definitions.
3. Run the script with `--yaml_path <your_config.yaml>` to load the new schema and automatically align the agent output with your PPT.

### Use custom templates

- Use the **Selection Pane** in PowerPoint to rename shapes so their names exactly match YAML field names (for example: `title`, `subtitle`, `item_title1`, `item_content1`, `label1`, `content1`).
- **No duplicate names** on the same slide. Each shape name must be unique so the renderer can target it deterministically.
- **Refer to existing templates** in `examples/ppt_gen/templates` and the field definitions in `yaml_example.yaml` as a naming guide.