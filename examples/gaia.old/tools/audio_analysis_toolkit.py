# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

# ruff: noqa

import base64
import os
from collections.abc import Callable
from urllib.parse import urlparse

import requests
from pydub.utils import mediainfo

from utu.config import ToolkitConfig
from utu.tools import AsyncBaseToolkit
from utu.utils import EnvUtils, SimplifiedAsyncOpenAI, get_logger

logger = get_logger(__name__)


class AudioAnalysisToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)

        self.cache_dir = "workspace/transcription_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.audio_client = SimplifiedAsyncOpenAI(
            api_key=EnvUtils.get_env("UTU_AUDIO_LLM_API_KEY"),
            base_url=EnvUtils.get_env("UTU_AUDIO_LLM_BASE_URL"),
        )
        self.audio_model = EnvUtils.get_env("UTU_AUDIO_LLM_MODEL")
        self.llm = SimplifiedAsyncOpenAI(**config.config_llm.model_provider.model_dump())

    def get_audio_duration(self, file_path):
        info = mediainfo(file_path)
        duration = float(info["duration"])
        return duration

    async def ask_question_about_audio(self, audio_path: str, question: str) -> str:
        r"""Ask any question about the audio and get the answer using
            multimodal model.

        Args:
            audio_path (str): The path to the audio file.
            question (str): The question to ask about the audio.

        Returns:
            str: The answer to the question.
        """

        logger.debug(
            f"Calling ask_question_about_audio method for audio file \
            `{audio_path}` and question `{question}`."
        )

        parsed_url = urlparse(audio_path)
        is_url = all([parsed_url.scheme, parsed_url.netloc])
        encoded_string = None

        if is_url:
            res = requests.get(audio_path)
            res.raise_for_status()
            audio_data = res.content
            encoded_string = base64.b64encode(audio_data).decode("utf-8")
        else:
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
            audio_file.close()
            encoded_string = base64.b64encode(audio_data).decode("utf-8")

        file_suffix = os.path.splitext(audio_path)[1]
        file_format = file_suffix[1:]

        # load cache if exists
        if os.path.exists(os.path.join(self.cache_dir, f"{os.path.basename(audio_path)}.txt")):
            logger.debug(f"Loading cached transcription for {audio_path}")
            with open(os.path.join(self.cache_dir, f"{os.path.basename(audio_path)}.txt")) as f:
                transcript = f.read().strip()
        else:
            transcription = await self.audio_client.audio.transcriptions.create(
                model=self.audio_model, file=open(audio_path, "rb")
            )

            transcript = transcription.text
            # save cache with audio file name
            cache_file = os.path.join(self.cache_dir, f"{os.path.basename(audio_path)}.txt")
            logger.debug(f"Saving transcription to cache: {cache_file}")
            with open(cache_file, "w") as f:
                f.write(transcript)

        reasoning_prompt = f"""
        <speech_transcription_result>{transcript}</speech_transcription_result>

        Please answer the following question based on the speech transcription result above:
        <question>{question}</question>
        """

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can answer questions about the given speech transcription.",
            },
            {"role": "user", "content": reasoning_prompt},
        ]

        # get the duration of the audio
        duration = self.get_audio_duration(audio_path)

        response: str = await self.llm.query_one(messages=messages, **self.config.config_llm.model_params.model_dump())
        response += f"\n\nAudio duration: {duration} seconds"

        logger.debug(f"Response: {response}")
        return response

    async def get_tools_map(self) -> dict[str, Callable]:
        return {
            "ask_question_about_audio": self.ask_question_about_audio,
        }
