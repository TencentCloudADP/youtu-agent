import asyncio

from utu.config import ConfigLoader
from utu.env import get_env
from utu.tools import FileEditToolkit, PythonExecutorToolkit
from utu.utils import DIR_ROOT


async def test_shell_env():
    agent_config = ConfigLoader.load_agent_config("examples/file_manager")
    async with await get_env(agent_config, "test_traceid") as env:
        # config = ConfigLoader.load_toolkit_config("bash")
        # config.config["workspace_root"] = str(DIR_ROOT / "data" / "test_bash")
        # bash_toolkit = BashToolkit(config=config)
        # bash_toolkit.setup_env(env)
        # result = await bash_toolkit.run_bash("pwd")
        # print(result)
        # result = await bash_toolkit.run_bash("wget https://www.gnu.org/software/wget/manual/wget.html -O wget.html")
        # print(result)

        config = ConfigLoader.load_toolkit_config("python_executor")
        config.config["workspace_root"] = str(DIR_ROOT / "data" / "test_bash")
        python_executor_toolkit = PythonExecutorToolkit(config=config)
        python_executor_toolkit.setup_env(env)
        result = await python_executor_toolkit.execute_python_code(code="import os\nos.getcwd()")
        print(result)

        config = ConfigLoader.load_toolkit_config("file_edit")
        config.config["workspace_root"] = str(DIR_ROOT / "data" / "test_bash")
        file_edit_toolkit = FileEditToolkit(config=config)
        file_edit_toolkit.setup_env(env)
        result = await file_edit_toolkit.write_file(path="test.txt", file_text="import os\nos.getcwd()")
        print(result)


if __name__ == "__main__":
    asyncio.run(test_shell_env())
