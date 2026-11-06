from pathlib import Path

import pytest

from utu.config import ToolkitConfig
from utu.tools.local_env.file_edit import FileEditLocal


def editor_at(tmp_path: Path, encoding: str) -> FileEditLocal:
    return FileEditLocal(ToolkitConfig(config={"workspace_root": str(tmp_path), "default_encoding": encoding}))


@pytest.mark.parametrize("encoding", ["utf-16", "gb18030", "latin-1", "utf-8"])
async def test_reads_files_using_the_configured_encoding(tmp_path: Path, encoding: str):
    content = "café\nsecond line\n"
    path = tmp_path / "input.txt"
    path.write_bytes(content.encode(encoding))

    assert await editor_at(tmp_path, encoding).read_file("input.txt") == content


@pytest.mark.parametrize("encoding", ["utf-16", "gb18030", "latin-1", "utf-8"])
async def test_written_files_can_be_read_and_edited_with_the_same_encoding(tmp_path: Path, encoding: str):
    editor = editor_at(tmp_path, encoding)
    content = "café\nsecond line\n"
    diff = "<<<<<<< SEARCH\nsecond line\n=======\nupdated line\n>>>>>>> REPLACE"

    await editor.write_file("nested/output.txt", content)
    path = tmp_path / "nested" / "output.txt"
    assert path.read_text(encoding=encoding) == content
    assert await editor.read_file("nested/output.txt") == content

    result = await editor.edit_file("nested/output.txt", diff)
    assert result.startswith("Successfully edited file:")
    expected = "café\nupdated line\n"
    assert path.read_text(encoding=encoding) == expected
    assert await editor.read_file("nested/output.txt") == expected
