import asyncio

from examples.gaia.tools.audio_analysis_toolkit import AudioAnalysisToolkit
from examples.gaia.tools.code_execution_toolkit import CodeExecutionToolkit
from examples.gaia.tools.document_processing_toolkit import DocumentProcessingToolkit
from examples.gaia.tools.excel_toolkit import ExcelToolkit
from examples.gaia.tools.image_analysis_toolkit import ImageAnalysisToolkit
from examples.gaia.tools.search_toolkit import SearchToolkit
from utu.config import ConfigLoader


async def test_excel_toolkit():
    excel_toolkit = ExcelToolkit()

    res = await excel_toolkit.extract_excel_content(
        "data/gaia/2023/validation/3da89939-209c-4086-8520-7eb734e6b4ef.xlsx"
    )
    print(res)


async def test_code_execution_toolkit():
    config = ConfigLoader.load_toolkit_config("arxiv")
    config.config = {"working_dir": "workspace/"}
    code_execution_toolkit = CodeExecutionToolkit(config)

    res = await code_execution_toolkit.execute_code("print('hello world')")
    print(f"> res: {res}")


async def test_image_analysis_toolkit():
    config = ConfigLoader.load_toolkit_config("image")
    image_analysis_toolkit = ImageAnalysisToolkit(config)

    res = await image_analysis_toolkit.image_to_text(
        "data/gaia/2023/validation/5b2a14e8-6e59-479c-80e3-4696e8980152.jpg"
    )
    print(f"> res: {res}")
    res = await image_analysis_toolkit.ask_question_about_image(
        "data/gaia/2023/validation/8f80e01c-1296-4371-9486-bb3d68651a60.png",
        "What is the name of the person in the image?",
    )
    print(f"> res: {res}")


async def test_audio_analysis_toolkit():
    config = ConfigLoader.load_toolkit_config("audio")
    audio_analysis_toolkit = AudioAnalysisToolkit(config)

    res = await audio_analysis_toolkit.ask_question_about_audio(
        "data/gaia/2023/validation/1f975693-876d-457b-a649-393859e79bf3.mp3",
        "What's the main topic of the audio?",
    )
    print(f"> res: {res}")


async def test_document_processing_toolkit():
    config = ConfigLoader.load_toolkit_config("arxiv")
    config.config = {"working_dir": "workspace/"}
    document_processing_toolkit = DocumentProcessingToolkit(config)

    tasks = (
        ("data/gaia/2023/validation/9b54f9d9-35ee-4a14-b62f-d130ea00317f.zip", None),
        ("data/gaia/2023/validation/67e8878b-5cef-4375-804e-e6291fdbe78a.pdf", None),
        ("data/gaia/2023/validation/1f975693-876d-457b-a649-393859e79bf3.mp3", None),
        ("data/gaia/2023/validation/3da89939-209c-4086-8520-7eb734e6b4ef.xlsx", None),
        ("data/gaia/2023/validation/8d46b8d6-b38a-47ff-ac74-cda14cf2d19b.csv", None),
        ("data/gaia/2023/validation/5b2a14e8-6e59-479c-80e3-4696e8980152.jpg", None),
        ("data/gaia/2023/validation/8f80e01c-1296-4371-9486-bb3d68651a60.png", None),
        ("data/gaia/2023/validation/a3fbeb63-0e8c-4a11-bff6-0e3b484c3e9c.pptx", None),
        ("data/gaia/2023/validation/389793a7-ca17-4e82-81cb-2b3a2391b4b9.txt", None),
        ("data/gaia/2023/validation/bec74516-02fc-48dc-b202-55e78d0e17cf.jsonld", None),
        ("data/gaia/2023/validation/f918266a-b3e0-4914-865d-4faa564f1aef.py", None),
        # ("data/gaia/2023/validation/7dd30055-0198-452e-8c25-f73dbe27dcb8.pdb", None),
    )

    for document_path, query in tasks:
        print(f"< document_path: {document_path} | query: {query}")
        res = await document_processing_toolkit.extract_document_content(document_path, query)
        print(f"> res: {res}")


async def test_search_toolkit():
    config = ConfigLoader.load_toolkit_config("arxiv")
    search_toolkit = SearchToolkit(config)

    res = await search_toolkit.search_google("What is the capital of France?")
    print(f"> res: {res}")
    res = await search_toolkit.multi_query_deep_search(["France", "capital France"], "What is the capital of France?")
    print(f"> res: {res}")
    res = await search_toolkit.parallel_search(["France", "capital France"], "What is the capital of France?")
    print(f"> res: {res}")


async def test_main():
    # await test_excel_toolkit()
    # await test_code_execution_toolkit()
    # await test_image_analysis_toolkit()
    # await test_audio_analysis_toolkit()
    # await test_document_processing_toolkit()
    await test_search_toolkit()


if __name__ == "__main__":
    asyncio.run(test_main())
