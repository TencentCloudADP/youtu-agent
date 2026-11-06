import pytest

from utu.utils import LLMOutputParser

I_CODE_BLOCK = """
这里是一些说明文本。

```python
def hello():
    print("Hello, world!")
```

```json
{"key": "value"}
```

```javascript
console.log("Hello, world!");
```
"""


def test_extract_code_blocks():
    code_blocks = LLMOutputParser.extract_code_blocks(I_CODE_BLOCK)
    print(code_blocks)


def test_extract_code_python():
    code = LLMOutputParser.extract_code_python(I_CODE_BLOCK)
    print(code)


def test_extract_code_json():
    code = LLMOutputParser.extract_code_json(I_CODE_BLOCK)
    print(code)
    code = LLMOutputParser.extract_code_json(I_CODE_BLOCK, try_parse=True)
    print(code)


def test_camel_to_snake():
    code = LLMOutputParser.camel_to_snake("ExtractVideoFrames")
    print(code)


def test_snake_to_camel():
    code = LLMOutputParser.snake_to_camel("extract_video_frames")
    print(code)
    code = LLMOutputParser.snake_to_camel("extract_video_frames", pascal=False)
    print(code)


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_extract_code_blocks_line_endings(newline):
    blocks = [
        ("python", f"def answer():{newline}    return 42"),
        ("json", '{"answer": 42}'),
        ("javascript", "console.log(42);"),
        ("yaml", "answer: 42"),
        ("", "untyped text"),
    ]
    text = f"Introduction{newline}" + newline.join(
        f"```{language}{newline}{code}{newline}```" for language, code in blocks
    )

    assert LLMOutputParser.extract_code_blocks(text) == blocks


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("language", ["py", "python", "PyThOn"])
def test_extract_code_python_line_endings(newline, language):
    expected = f"def generated(value):{newline}    return value + 1"
    text = f'```json{newline}{{"ignored": true}}{newline}```{newline}```{language}{newline}{expected}{newline}```'

    code = LLMOutputParser.extract_code_python(text)

    assert code == expected
    compile(code, "generated_runner.py", "exec")


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("try_parse", [False, True])
def test_extract_code_json_line_endings(newline, try_parse):
    payload = '{"name": "demo", "requirements": [], "enabled": true}'
    text = f"```python{newline}print('not json'){newline}```{newline}```json{newline}{payload}{newline}```"

    result = LLMOutputParser.extract_code_json(text, try_parse=try_parse)

    expected = {"name": "demo", "requirements": [], "enabled": True} if try_parse else payload
    assert result == expected


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize(
    ("language", "tag", "code"),
    [("javascript", "JS", "console.log(42);"), ("yaml", "YML", "answer: 42")],
)
def test_extract_code_block_language_aliases(newline, language, tag, code):
    text = f"```python{newline}pass{newline}```{newline}```{tag}{newline}{code}{newline}```"

    assert LLMOutputParser.extract_code_block_with_language(text, language) == code


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_extract_code_block_missing_language_uses_first(newline):
    text = f"```yaml{newline}first: true{newline}```{newline}```json{newline}{{}}{newline}```"

    assert LLMOutputParser.extract_code_python(text) == "first: true"


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_extract_code_json_invalid_json(newline):
    code = '{"items": [}'

    assert LLMOutputParser.extract_code_json(f"```json{newline}{code}{newline}```") == code


def test_extract_code_python_without_fences_preserves_input():
    code = "\nprint('unfenced')\n"

    assert LLMOutputParser.extract_code_python(code) == code


@pytest.mark.parametrize("try_parse", [False, True])
def test_extract_code_json_without_fences(try_parse):
    code = '{"answer": 42}'

    assert LLMOutputParser.extract_code_json(code, try_parse=try_parse) == ({"answer": 42} if try_parse else code)


def test_extract_invalid_json_without_fences_preserves_input():
    code = "  not json  "

    assert LLMOutputParser.extract_code_json(code) == code


@pytest.mark.parametrize(
    ("name", "expected"),
    [("ExtractVideoFrames", "extract_video_frames"), ("HTTPRequest", "http_request")],
)
def test_camel_to_snake_neighbor(name, expected):
    assert LLMOutputParser.camel_to_snake(name) == expected


@pytest.mark.parametrize(("pascal", "expected"), [(True, "ExtractVideoFrames"), (False, "extractVideoFrames")])
def test_snake_to_camel_neighbor(pascal, expected):
    assert LLMOutputParser.snake_to_camel("extract_video_frames", pascal=pascal) == expected


@pytest.mark.parametrize(("return_last", "expected"), [(False, 1.25), (True, 3.5)])
def test_extract_float_number_neighbor(return_last, expected):
    assert LLMOutputParser.extract_float_number("first=1.25, last=3.5", return_last=return_last) == expected
