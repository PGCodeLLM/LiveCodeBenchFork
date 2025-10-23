import os
import json
import time
from time import sleep

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from lcb_runner.lm_styles import LMStyle
from lcb_runner.runner.base_runner import BaseRunner


class OpenAIRunner(BaseRunner):
    client = OpenAI()

    def __init__(self, args, model):
        super().__init__(args, model)

        # Parse user-provided extra_body and extra_headers
        extra_body = json.loads(args.extra_body) if args.extra_body else {}
        extra_headers = json.loads(args.extra_headers) if args.extra_headers else {}

        # Store stream setting
        self.stream = getattr(args, 'stream', False)
        if self.stream:
            print("[LCB] Streaming is enabled.")

        if model.model_style == LMStyle.OpenAIReasonPreview:
            self.client_kwargs = {
                "model": args.model,
                "max_completion_tokens": 25000,
            }
        elif model.model_style == LMStyle.OpenAIReason:
            assert (
                "__" in args.model
            ), f"Model {args.model} is not a valid OpenAI Reasoning model as we require reasoning effort in model name."
            model, reasoning_effort = args.model.split("__")
            self.client_kwargs = {
                "model": model,
                "reasoning_effort": reasoning_effort,
            }
        else:
            if args.top_k is not None:
                extra_body["top_k"] = args.top_k
            if args.repetition_penalty is not None:
                extra_body["repetition_penalty"] = args.repetition_penalty

            self.client_kwargs = {
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "n": args.n,
                "timeout": args.openai_timeout,
                # "stop": args.stop, --> stop is only used for base models currently
            }
            if args.max_tokens is not None:
                self.client_kwargs["max_tokens"] = args.max_tokens

            if args.presence_penalty is not None:
                self.client_kwargs["presence_penalty"] = args.presence_penalty

            # Only add extra_body/extra_headers if they have content
            if extra_body:
                self.client_kwargs["extra_body"] = extra_body
            if extra_headers:
                self.client_kwargs["extra_headers"] = extra_headers

        print(f"[LCB Inference Parameters] {self.client_kwargs}")

    def _run_single(self, prompt: list[dict[str, str]], n: int = 10) -> list[str]:
        assert isinstance(prompt, list)

        if n == 0:
            print("Max retries reached. Returning empty response.")
            return []

        try:
            if not self.stream:
                response = OpenAIRunner.client.chat.completions.create(
                    messages=prompt,
                    **self.client_kwargs,
                )
            else:
                # Handle streaming
                stream_response = OpenAIRunner.client.chat.completions.create(
                    messages=prompt,
                    stream=True,
                    **self.client_kwargs,
                )

                num_samples = self.client_kwargs.get("n", 1)
                accumulated_content = [""] * num_samples
                finish_reasons = [None] * num_samples
                roles = [None] * num_samples

                for chunk in stream_response:
                    if not chunk.choices:
                        continue
                    for choice in chunk.choices:
                        idx = choice.index
                        if choice.delta:
                            if choice.delta.content:
                                accumulated_content[idx] += choice.delta.content
                            if choice.delta.role:
                                roles[idx] = choice.delta.role
                        if choice.finish_reason:
                            finish_reasons[idx] = choice.finish_reason

                choices = []
                for i in range(num_samples):
                    role = roles[i] if roles[i] else "assistant"
                    finish_reason = finish_reasons[i] if finish_reasons[i] else "stop"
                    choices.append(
                        Choice(
                            index=i,
                            finish_reason=finish_reason,
                            message=ChatCompletionMessage(
                                role=role,
                                content=accumulated_content[i],
                            ),
                        )
                    )

                response = ChatCompletion(
                    id="streamed-" + str(int(time.time())),
                    object="chat.completion",
                    created=int(time.time()),
                    model=self.client_kwargs.get("model", "unknown"),
                    choices=choices,
                )

        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return self._run_single(prompt, n=n - 1)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
        return [c.message.content or "" for c in response.choices]
