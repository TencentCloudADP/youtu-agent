"""
https://github.com/pexpect/pexpect
@ii-agent/src/ii_agent/tools/bash_tool.py


--- https://www.anthropic.com/engineering/swe-bench-sonnet ---
Run commands in a bash shell\n
* When invoking this tool, the contents of the \"command\" parameter does NOT need to be XML-escaped.\n
* You don't have access to the internet via this tool.\n
* You do have access to a mirror of common linux and python packages via apt and pip.\n
* State is persistent across command calls and discussions with the user.\n
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.\n
* Please avoid commands that may produce a very large amount of output.\n
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background."
"""

import pathlib
import re
import sys

import pexpect

from ..config import ToolkitConfig
from ..utils import get_logger
from .base import AsyncBaseToolkit, register_tool

logger = get_logger(__name__)


class BashToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)
        # self.require_confirmation = self.config.config.get("require_confirmation", False)
        # self.command_filters = self.config.config.get("command_filters", [])
        self.timeout = self.config.config.get("timeout", 60)
        self.banned_command_strs = [
            "git init",
            "git commit",
            "git add",
        ]

        self.child, self.custom_prompt = self.start_persistent_shell(timeout=self.timeout)

        workspace_root = self.config.config.get("workspace_root", "/tmp/")
        self.setup_workspace(workspace_root)

    def setup_workspace(self, workspace_root: str):
        workspace_dir = pathlib.Path(workspace_root)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_root = workspace_root
        self.run_command(self.child, self.custom_prompt, f"cd {workspace_root}")

    @staticmethod
    def start_persistent_shell(timeout: int):
        # https://github.com/pexpect/pexpect/issues/321

        # Start a new Bash shell
        if sys.platform == "win32":
            child = pexpect.spawn("cmd.exe", encoding="utf-8", echo=False, timeout=timeout)
            custom_prompt = "PROMPT_>"
            child.sendline(f"prompt {custom_prompt}")
            child.expect(custom_prompt)
        else:
            child = pexpect.spawn("/bin/bash", encoding="utf-8", echo=False, timeout=timeout)
            # Set a known, unique prompt
            # We use a random string that is unlikely to appear otherwise
            # so we can detect the prompt reliably.
            custom_prompt = "PEXPECT_PROMPT>> "
            child.sendline("stty -onlcr")
            child.sendline("unset PROMPT_COMMAND")
            child.sendline(f"PS1='{custom_prompt}'")
            # Force an initial read until the newly set prompt shows up
            child.expect(custom_prompt)
            return child, custom_prompt

    @staticmethod
    def run_command(child, custom_prompt: str, cmd: str) -> str:
        # Send the command
        child.sendline(cmd)
        # Wait until we see the prompt again
        child.expect(custom_prompt)
        # Output is everything printed before the prompt minus the command itself
        # pexpect puts the matched prompt in child.after and everything before it in child.before.

        raw_output = child.before.strip()
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        clean_output = ansi_escape.sub("", raw_output)

        if clean_output.startswith("\r"):
            clean_output = clean_output[1:]

        return clean_output

    @register_tool
    async def run_bash(self, command: str) -> str:
        """Execute a bash command in your workspace and return its output.

        Args:
            command: The command to execute
        """
        # 1) filter: change command before execution. E.g. used in SSH or Docker.
        # original_command = command
        # command = self.apply_filters(original_command)
        # if command != original_command:
        #     logger.info(f"Command filtered: {original_command} -> {command}")

        # 2) banned command check
        for banned_str in self.banned_command_strs:
            if banned_str in command:
                return f"Command not executed due to banned string in command: {banned_str} found in {command}."

        # if self.require_confirmation:
        #     ...

        # confirm no bad stuff happened
        try:
            echo_result = self.run_command(self.child, self.custom_prompt, "echo hello")
            assert echo_result.strip() == "hello"
        except Exception:  # pylint: disable=broad-except
            self.child, self.custom_prompt = self.start_persistent_shell(self.timeout)

        # 3) Execute the command and capture output
        try:
            result = self.run_command(self.child, self.custom_prompt, command)
            return str(
                {
                    "command output": result,
                }
            )
        except Exception as e:  # pylint: disable=broad-except
            return str(
                {
                    "error": str(e),
                }
            )
        # TODO: add workspace tree in output
