import os

from ...utils import DIR_ROOT
from ..data import EvaluationSample
from .base_llm_processor import BaseLLMJudgeProcesser
from .prompts import AUGMENTATION_PROMPTS


class GAIAProcesser(BaseLLMJudgeProcesser):
    """Processer for GAIA evaluation."""

    name: str = "GAIA"

    def preprocess_one(self, sample: EvaluationSample, recorder=None) -> EvaluationSample:
        """Preprocess a single sample."""
        _ = recorder
        question_with_files = sample.raw_question + self._get_file_prompt(sample.file_name)
        augmented_question = AUGMENTATION_PROMPTS["GAIA"].format(question=question_with_files)
        sample.update(
            augmented_question=augmented_question,
        )
        return sample

    def _get_file_prompt(self, file_name: str) -> str:
        """
        Get the file prompt if applicable.
        """
        prompt_use_files = ""
        file_name = self._formulate_file_path(file_name)
        if file_name:
            if ".MOV" in file_name:
                return ""
            prompt_use_files += "\n\nTo answer the question above, you will have to use these attached files:"
            if file_name.split(".")[-1] in ["pdf", "xlsx"]:
                image_path = file_name.split(".")[0] + ".png"
                if os.path.exists(image_path):
                    prompt_use_files += f"\nAttached image: {image_path}"
                else:
                    prompt_use_files += f"\nAttached file: {file_name}"
            elif file_name.split(".")[-1] == "zip":
                import shutil

                folder_name = file_name.replace(".zip", "")
                os.makedirs(folder_name, exist_ok=True)
                shutil.unpack_archive(file_name, folder_name)

                # Convert the extracted files
                prompt_use_files = (
                    "\n\nYou have been given a zip archive of supporting files. We extracted it into a directory: "
                    "find the extracted files at the following paths:\n"
                )
                for root, _, files in os.walk(folder_name):
                    for file in files:
                        file_path = os.path.join(root, file)
                        prompt_use_files += f"- {file_path}\n"
            elif file_name.split(".")[-1] in ["png", "jpg", "jpeg"]:
                prompt_use_files += f"\nAttached image: {file_name}"
            elif file_name.split(".")[-1] in ["mp3", "m4a", "wav"]:
                prompt_use_files += f"\nAttached audio: {file_name}"
            else:
                prompt_use_files += f"\nAttached file: {file_name}"
        else:
            prompt_use_files += "\n\nYou have been given no local files to access."

        return prompt_use_files

    def _formulate_file_path(self, file_name: str) -> str:
        """Formulate the file name to the absolute path of the file."""
        if not file_name:
            return ""
        split = self.config.data.dataset.split("_")[-1]  # GAIA_validation
        dataset_dir = DIR_ROOT / "data" / "gaia" / "2023" / split
        attach_file_path = dataset_dir / file_name
        return str(attach_file_path)
