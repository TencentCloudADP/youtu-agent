import time

import openai

from utu.utils import EnvUtils


class LLM:
    def __init__(self):
        EnvUtils.assert_env(["UTU_LLM_TYPE", "UTU_LLM_MODEL", "UTU_LLM_BASE_URL", "UTU_LLM_API_KEY"])
        self.model_name = EnvUtils.get_env("UTU_LLM_MODEL")
        self.client = openai.OpenAI(
            api_key=EnvUtils.get_env("UTU_LLM_API_KEY"),
            base_url=EnvUtils.get_env("UTU_LLM_BASE_URL"),
        )

    def chat(self, messages_or_prompt, max_tokens=16384, temperature=0, max_retries=3, return_reasoning=False):
        for _ in range(max_retries):
            try:
                if isinstance(messages_or_prompt, str):
                    messages = [{"role": "user", "content": messages_or_prompt}]
                elif isinstance(messages_or_prompt, list):
                    messages = messages_or_prompt
                else:
                    raise ValueError("messages_or_prompt must be a string or a list of messages.")

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                response_text = response.choices[0].message.content.strip()

                if return_reasoning:
                    reasoning = response.choices[0].message.reasoning_content
                    return response_text, reasoning
                return response_text

            except Exception as e:
                error = f"An unexpected error occurred: {e}"
                print(error)
            time.sleep(10)
