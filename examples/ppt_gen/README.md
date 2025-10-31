# PowerPoint Template Example

## Quick Start

Download [template files](https://drive.google.com/drive/folders/1oZYoHBJJwzAzhrpvnUnklvU7LcS4oFz6?usp=sharing) to `examples/ppt_gen/template` directory.

Prepare the reference resource (plain text / markdown / html webpage) for PPT generation. For example, download the [Nobel Prize webpage](https://www.nobelprize.org/prizes/physics/2025/popular-information/).

```
wget https://www.nobelprize.org/prizes/physics/2025/popular-information/ -O webpage.html
```

`cd` into `examples/ppt_gen`

Run `main.py`

```
python main.py --file webpage.html --template_path template/0.pptx --pages 15
```

The script will produce a `json` file and a `pptx` file if everything is OK.

## Custom Template

The PPT generation workflow basically consists of 3 stages

1. LLM summarize the give resource, and arrange the content for PPT
2. LLM rearrange the content into JSON format following the given JSON schema which defines various types of layout and content requirements.
3. Extract JSON from LLM output, fill the PPT template by duplicating slides and replacing shapes in the template file.

The third stage is rigid and fixed, so custom templates must be pre-labeled consistently with the definitions in the schema.

First, prepare a template by referring to existing templates, and arrange the following types of pages in order:

1. Single-column content page (content)
2. Title page (title)
3. Section title page (section_title)
4. Double-column content page
5. Bullet points (3 items)
6. Bullet points (4 items)
7. Bullet points (4 items)
8. Bullet points (4 items, showing upward trend)
9. Bullet points (4 items, with single-word headings)
10. Bullet points (6 items, displaying process/timeline)
11. Bullet points (4 items, wrapping around an image)
12. Bullet points (5 items, displaying process/timeline)
13. Acknowledgments page

Then, for each page, enter the *selection pane* and rename specific elements so that their names correspond exactly to the requirements specified in the schema. You may refer to the existing template, JSON schmea and the Python code for proper labeling.

JSON schema can be customized too. The script `synthesis_config.py` can be a handy tool to update agent config file.
