import pytest

from utu.config import ConfigLoader
from utu.tools import DocumentToolkit


@pytest.fixture
def document_toolkit() -> DocumentToolkit:
    config = ConfigLoader.load_toolkit_config("document")
    return DocumentToolkit(config=config)


async def test_document_toolkit(document_toolkit: DocumentToolkit):
    q = "There is a diagram of an X-ray time profile in this document. How long is the time profile (in seconds)?"
    result = await document_toolkit.document_qa(document_path="https://arxiv.org/pdf/2107.14339.pdf", question=q)
    print(result)
    result = await document_toolkit.document_qa(document_path="https://arxiv.org/pdf/2107.14339.pdf")
    print(result)
