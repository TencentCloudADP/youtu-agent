import time
from utu.utils import EnvUtils

try:
    import openai  # kept for OpenAI/HTTP engine
except Exception:
    openai = None

class LLM:
    def __init__(self):
        # Default logic: if weights are provided, prefer native vLLM; otherwise HTTP(OpenAI)
        engine = EnvUtils.get_env("UTU_LLM_ENGINE", required=False)
        if not engine:
            engine = "vllm" if EnvUtils.get_env("UTU_LLM_WEIGHTS", required=False) else "openai"
        self.engine = engine.lower()

        if self.engine == "vllm":
            # Native vLLM backend (no HTTP server required)
            weights_path = EnvUtils.get_env("UTU_LLM_WEIGHTS", required=False)
            if not weights_path:
                raise ValueError("UTU_LLM_WEIGHTS is required when UTU_LLM_ENGINE=vllm")
            try:
                from vllm import LLM as VLLMEngine
                from vllm import SamplingParams
            except Exception as e:
                raise ImportError("vllm is required for UTU_LLM_ENGINE=vllm. Please: pip install vllm") from e
            self._vllm_engine = VLLMEngine(model=weights_path)
            self._vllm_sampling_cls = SamplingParams
        else:
            # Default OpenAI-compatible HTTP engine
            EnvUtils.assert_env(["UTU_LLM_TYPE", "UTU_LLM_MODEL", "UTU_LLM_BASE_URL", "UTU_LLM_API_KEY"])
            self.model_name = EnvUtils.get_env("UTU_LLM_MODEL")
            if openai is None:
                raise ImportError("openai package is required for HTTP engine. Ensure it's installed.")
            self.client = openai.OpenAI(
                api_key=EnvUtils.get_env("UTU_LLM_API_KEY"),
                base_url=EnvUtils.get_env("UTU_LLM_BASE_URL"),
            )

    def chat(
        self,
        messages_or_prompt,
        max_tokens=16384,
        temperature=0,
        top_p: float | None = None,
        top_k: int | None = None,
        max_retries=3,
        return_reasoning=False,
    ):
        # vLLM native
        if self.engine == "vllm":
            if isinstance(messages_or_prompt, str):
                prompt = messages_or_prompt
            elif isinstance(messages_or_prompt, list):
                # Simple conversion of messages to a single prompt
                prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages_or_prompt)
            else:
                raise ValueError("messages_or_prompt must be a string or a list of messages.")

            params = {"temperature": temperature, "max_tokens": max_tokens}
            if top_p is not None:
                params["top_p"] = top_p
            if top_k is not None:
                params["top_k"] = top_k
            sp = self._vllm_sampling_cls(**params)
            outputs = self._vllm_engine.generate([prompt], sp)
            text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            return text.strip()

        # OpenAI-compatible HTTP
        for _ in range(max_retries):
            try:
                if isinstance(messages_or_prompt, str):
                    messages = [{"role": "user", "content": messages_or_prompt}]
                elif isinstance(messages_or_prompt, list):
                    messages = messages_or_prompt
                else:
                    raise ValueError("messages_or_prompt must be a string or a list of messages.")

                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if top_p is not None:
                    kwargs["top_p"] = top_p
                # Non-standard, but many servers accept it
                if top_k is not None:
                    kwargs["extra_body"] = {"top_k": top_k}
                response = self.client.chat.completions.create(**kwargs)
                response_text = response.choices[0].message.content.strip()

                if return_reasoning:
                    reasoning = getattr(response.choices[0].message, "reasoning_content", None)
                    return response_text, reasoning
                return response_text

            except Exception as e:
                print(f"LLM.chat error: {e}")
            time.sleep(10)