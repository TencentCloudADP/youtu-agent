import json

from utu.config import ConfigLoader


def test_load_subconfig():
    config = ConfigLoader._load_config_to_dict("tools/search")
    print(config)


def test_load_model_config():
    config = ConfigLoader.load_model_config("base")
    print(config)


def test_load_toolkit_config():
    config = ConfigLoader.load_toolkit_config("search")
    config = ConfigLoader.load_toolkit_config("document")
    config = ConfigLoader.load_toolkit_config("python_executor")
    config = ConfigLoader.load_toolkit_config("generated/download_bilibili_video")
    print(config)


def test_load_agent_config():
    config = ConfigLoader.load_agent_config("simple/search_agent")
    config = ConfigLoader.load_agent_config("simple/gaia_reasoning_coding")
    config = ConfigLoader.load_agent_config("simple/gaia_document_processing")
    config = ConfigLoader.load_agent_config("simple/gaia_web_search")
    config = ConfigLoader.load_agent_config("simple/base_search")
    config = ConfigLoader.load_agent_config("workforce/base")
    config = ConfigLoader.load_agent_config("examples/data_analysis")
    config = ConfigLoader.load_agent_config("examples/file_manager")
    config = ConfigLoader.load_agent_config("examples/svg_generator")
    config = ConfigLoader.load_agent_config("examples/paper_collector")
    config = ConfigLoader.load_agent_config("orchestrator/base_test")
    print(config)
    # print(json.dumps(config.model_dump(), indent=2))


def test_load_eval_config():
    config = ConfigLoader.load_eval_config("ww")
    config = ConfigLoader.load_eval_config("gaia")
    print(json.dumps(config.model_dump(), indent=2))
