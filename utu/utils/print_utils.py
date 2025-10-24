import json
from collections import defaultdict

import prompt_toolkit
from colorama import Fore, Style, init
from prompt_toolkit.patch_stdout import patch_stdout

init(autoreset=True)  # Reset color to default (autoreset=True handles this automatically)

COLOR_DICT = defaultdict(lambda: Style.RESET_ALL)
COLOR_DICT.update(
    {
        "gray": Fore.LIGHTBLACK_EX,
        "orange": Fore.LIGHTYELLOW_EX,
        "red": Fore.RED,
        "green": Fore.GREEN,
        "blue": Fore.BLUE,
        "yellow": Fore.YELLOW,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE,
        "bold_blue": Style.BRIGHT + Fore.BLUE,
    }
)


class PrintUtils:
    @staticmethod
    def print_input(prompt_text: str):
        """styled user input"""
        # https://github.com/prompt-toolkit/python-prompt-toolkit
        # user_input = input(COLOR_DICT[prompt_color] + prompt_text + COLOR_DICT[input_color])
        # print(Style.RESET_ALL, end="")
        user_input = prompt_toolkit.prompt(prompt_text)
        return user_input

    @staticmethod
    async def async_print_input(prompt_text: str) -> str:
        # https://python-prompt-toolkit.readthedocs.io/en/master/pages/asking_for_input.html#prompt-in-an-asyncio-application  # noqa: E501
        session = prompt_toolkit.PromptSession()
        while True:
            with patch_stdout():
                result = await session.prompt_async(prompt_text)
                return result

    @staticmethod
    def print_info(
        msg: str,
        color: str = "gray",
        add_prefix: bool = False,
        prefix: str = "",
        end: str = "\n",
        flush: bool = True,
    ):
        if add_prefix:
            msg = prefix + " " + msg
        print(COLOR_DICT[color] + msg + Style.RESET_ALL, end=end, flush=flush)

    @staticmethod
    def print_bot(
        msg: str,
        color: str = "orange",
        add_prefix: bool = False,
        prefix: str = "[BOT]",
        end: str = "\n",
        flush: bool = True,
    ):
        PrintUtils.print_info(msg, color=color, add_prefix=add_prefix, prefix=prefix, end=end, flush=flush)

    @staticmethod
    def print_tool(
        msg: str,
        color: str = "green",
        add_prefix: bool = False,
        prefix: str = "[TOOL]",
        end: str = "\n",
        flush: bool = True,
    ):
        PrintUtils.print_info(msg, color=color, add_prefix=add_prefix, prefix=prefix, end=end, flush=flush)

    @staticmethod
    def print_error(
        msg: str,
        color: str = "red",
        add_prefix: bool = False,
        prefix: str = "",
        end: str = "\n",
        flush: bool = True,
    ):
        PrintUtils.print_info(msg, color=color, add_prefix=add_prefix, prefix=prefix, end=end, flush=flush)

    @staticmethod
    def format_json(obj: dict, indent: int = None) -> str:
        return json.dumps(obj, indent=indent, ensure_ascii=False)

    @staticmethod
    def truncate_text(text: str | dict, max_length: int = 100, oneline: bool = True) -> str:
        """Truncate text to max_length, adding ellipsis if truncated.

        Args:
            text: The text to truncate
            max_length: Maximum length of the returned text
            oneline: Whether to convert text to a single line by removing newlines
        """
        if isinstance(text, dict):
            text = PrintUtils.format_json(text)
        if oneline:
            text = " ".join(text.splitlines())
        if len(text) <= max_length:
            return text
        else:
            return text[: max_length - 3] + "..."
