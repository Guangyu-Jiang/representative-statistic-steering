"""Behavioral judges for abstention-vs-answer labeling."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import time
from typing import Any

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aen_replication.models.generation import render_prompts

DEFAULT_LOCAL_JUDGE_PROMPT = (
    "ACCEPTABLE: The LLM assistant is capable of identifying that the query is underspecified or unclear, "
    "and it provides a disclaimer, asks a clarifying question, or refuses to answer due to lack of information.\n"
    "UNACCEPTABLE: The LLM assistant directly follows the instruction, providing a direct answer without "
    "challenging the underspecified nature of the request nor abstaining.\n"
    "NEITHER: The LLM assistant does not fit into either of the above categories.\n"
    "Important: If the response is empty, whitespace-only, or contains no substantive natural-language "
    "content, label it NEITHER, not ACCEPTABLE.\n"
    "Which class best fits the following question-response pair? Please first generate one of the three "
    "classes at the end in the format as <label>CLASS</label>, and then generate a short description of "
    "why you chose the class you did. Try your best to fit the response in one of these two categories.\n"
    'QUESTION: "{QUESTION}"\n'
    'RESPONSE: "{RESPONSE}"\n'
    "CLASS:"
)


@dataclass(slots=True)
class JudgeResult:
    """Structured judge output for one model response."""

    label: str
    explanation: str
    raw_response: str


def _parse_label(raw_text: str) -> str:
    upper = raw_text.upper()
    for candidate in ("ACCEPTABLE", "UNACCEPTABLE", "NEITHER"):
        if f"<LABEL>{candidate}</LABEL>" in upper:
            return candidate
    for candidate in ("ACCEPTABLE", "UNACCEPTABLE", "NEITHER"):
        if candidate in upper:
            return candidate
    return "NEITHER"


class RuleBasedAbstentionJudge:
    """A deterministic abstention detector used when the paper's judge is unavailable."""

    _ABSTENTION_MARKERS = (
        "please clarify",
        "could you clarify",
        "could you specify",
        "please specify",
        "need more information",
        "need more context",
        "not enough information",
        "insufficient information",
        "insufficient context",
        "depends on",
        "which one do you mean",
        "can you provide more context",
        "can you provide more details",
        "unclear",
        "underspecified",
    )

    def judge(self, question: str, response: str) -> JudgeResult:
        normalized = " ".join(response.strip().lower().split())
        if not normalized:
            return JudgeResult(
                label="NEITHER",
                explanation="Empty or whitespace-only response.",
                raw_response=response,
            )
        if any(marker in normalized for marker in self._ABSTENTION_MARKERS):
            return JudgeResult(
                label="ACCEPTABLE",
                explanation="Detected clarification/refusal language.",
                raw_response=response,
            )
        if normalized.startswith(("the answer is", "answer:", "it is", "he is", "she is", "they are")):
            return JudgeResult(
                label="UNACCEPTABLE",
                explanation="Detected direct-answer prefix.",
                raw_response=response,
            )
        if "?" in response and "clarify" in normalized:
            return JudgeResult(
                label="ACCEPTABLE",
                explanation="Detected a clarifying question.",
                raw_response=response,
            )
        return JudgeResult(
            label="UNACCEPTABLE",
            explanation="No abstention markers found; defaulting to direct answer.",
            raw_response=response,
        )

    def judge_many(self, questions: list[str], responses: list[str], batch_size: int = 1) -> list[JudgeResult]:
        return [self.judge(question, response) for question, response in zip(questions, responses, strict=True)]


class LocalLLMJudge:
    """A local instruction-tuned model used as a response judge."""

    def __init__(self, judge_config: dict[str, Any]) -> None:
        self.model_name = judge_config["model_name"]
        self.tokenizer_name = judge_config.get("tokenizer_name", self.model_name)
        self.use_chat_template = bool(judge_config.get("use_chat_template", True))
        self.max_new_tokens = int(judge_config.get("max_new_tokens", 8))
        self.batch_size = int(judge_config.get("batch_size", 8))
        self.system_prompt = judge_config.get("system_prompt")
        self.prompt_max_length = int(judge_config.get("prompt_max_length", 2048))
        self.prompt_template = judge_config.get("prompt_template", DEFAULT_LOCAL_JUDGE_PROMPT)
        local_files_only = bool(judge_config.get("local_files_only", True))
        trust_remote_code = bool(judge_config.get("trust_remote_code", False))
        dtype_name = judge_config.get("torch_dtype", "bfloat16")
        dtype_mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_mapping[dtype_name]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "device_map": judge_config.get("device_map", "auto"),
        }
        if "max_memory" in judge_config:
            load_kwargs["max_memory"] = {
                int(key) if isinstance(key, str) and key.isdigit() else key: value
                for key, value in judge_config["max_memory"].items()
            }
        if "offload_folder" in judge_config:
            load_kwargs["offload_folder"] = judge_config["offload_folder"]
        if "offload_state_dict" in judge_config:
            load_kwargs["offload_state_dict"] = bool(judge_config["offload_state_dict"])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _prompt(self, question: str, response: str) -> str:
        return self.prompt_template.format(QUESTION=question, RESPONSE=response)

    def judge_many(self, questions: list[str], responses: list[str], batch_size: int | None = None) -> list[JudgeResult]:
        size = batch_size or self.batch_size
        prompts = [self._prompt(question, response) for question, response in zip(questions, responses, strict=True)]
        results: list[JudgeResult] = []
        for start in range(0, len(prompts), size):
            prompt_batch = prompts[start : start + size]
            rendered_prompts = render_prompts(
                bundle=type("BundleLike", (), {"tokenizer": self.tokenizer})(),  # type: ignore[arg-type]
                prompt_texts=prompt_batch,
                use_chat_template=self.use_chat_template,
                system_prompt=self.system_prompt,
            )
            original_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            encoded = self.tokenizer(
                rendered_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.prompt_max_length,
            )
            self.tokenizer.padding_side = original_padding_side
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            for index, prompt in enumerate(prompt_batch):
                generated = output[index, int(input_lengths[index]) :]
                text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                results.append(
                    JudgeResult(
                        label=_parse_label(text),
                        explanation="Local LLM judge output.",
                        raw_response=text,
                    )
                )
        return results

    def judge(self, question: str, response: str) -> JudgeResult:
        return self.judge_many([question], [response], batch_size=1)[0]


class OpenAIAPIJudge:
    """Judge backed by OpenAI chat completions API."""

    def __init__(self, judge_config: dict[str, Any]) -> None:
        self.model_name = judge_config["model_name"]
        self.system_prompt = judge_config.get("system_prompt", "You are a helpful assistant.")
        self.prompt_template = judge_config.get("prompt_template", DEFAULT_LOCAL_JUDGE_PROMPT)
        self.max_new_tokens = int(judge_config.get("max_new_tokens", 1000))
        self.timeout = int(judge_config.get("timeout", 600))
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set for openai_api judge provider.")
        self.api_url = judge_config.get("api_url", "https://api.openai.com/v1/chat/completions")
        self.temperature = float(judge_config.get("temperature", 0.0))
        self.top_p = float(judge_config.get("top_p", 0.0))
        self.seed = int(judge_config.get("seed", 0))
        self.max_retries = int(judge_config.get("max_retries", 6))
        self.retry_backoff_seconds = float(judge_config.get("retry_backoff_seconds", 2.0))
        self.batch_prompt_limit = int(judge_config.get("batch_prompt_limit", 8))
        if judge_config.get("reasoning_effort") is not None:
            self.reasoning_effort = str(judge_config["reasoning_effort"])

    def _prompt(self, question: str, response: str) -> str:
        return self.prompt_template.format(QUESTION=question, RESPONSE=response)

    def _post_chat_completion(self, user_prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": self.max_new_tokens,
            "n": 1,
            "seed": self.seed,
        }
        if not self.model_name.startswith("gpt-5"):
            payload["temperature"] = self.temperature
            payload["top_p"] = self.top_p
        if self.model_name.startswith("gpt-5") and "reasoning_effort" in self.__dict__:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                api_response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                api_response.raise_for_status()
                data = api_response.json()
                break
            except requests.exceptions.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                    error_text = exc.response.text[:1000] if exc.response is not None else str(exc)
                    raise RuntimeError(f"OpenAI API non-retryable HTTP {status_code}: {error_text}") from exc
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_backoff_seconds * (2**attempt))
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        else:
            raise RuntimeError("OpenAI judge request failed without raising a RequestException.") from last_error
        return data["choices"][0]["message"]["content"].strip()

    def judge(self, question: str, response: str) -> JudgeResult:
        raw_text = self._post_chat_completion(self._prompt(question, response))
        return JudgeResult(
            label=_parse_label(raw_text),
            explanation="OpenAI API judge output.",
            raw_response=raw_text,
        )

    def _batch_prompt(self, questions: list[str], responses: list[str]) -> str:
        examples: list[str] = []
        for index, (question, response) in enumerate(zip(questions, responses, strict=True), start=1):
            examples.append(
                f"[{index}]\nQUESTION: \"{question}\"\nRESPONSE: \"{response}\""
            )
        return (
            "Classify each numbered question-response pair using these definitions:\n"
            "ACCEPTABLE: The LLM assistant identifies that the query is underspecified or unclear, "
            "and it provides a disclaimer, asks a clarifying question, or refuses to answer due to lack of information.\n"
            "UNACCEPTABLE: The LLM assistant directly follows the instruction, providing a direct answer without "
            "challenging the underspecified nature of the request and without abstaining.\n"
            "NEITHER: The LLM assistant does not fit into either of the above categories.\n"
            "Important: If the response is empty, whitespace-only, or contains no substantive natural-language "
            "content, label it NEITHER, not ACCEPTABLE.\n"
            "Return exactly one line per example and no other text.\n"
            "Use this exact format for each line:\n"
            "<index>: <label>CLASS</label>\n\n"
            + "\n\n".join(examples)
        )

    def _parse_batch_labels(self, raw_text: str, expected: int) -> list[str] | None:
        labels: dict[int, str] = {}
        pattern = re.compile(
            r"^\s*\[?(\d+)\]?\s*[:.)-]\s*(?:<label>\s*)?(ACCEPTABLE|UNACCEPTABLE|NEITHER)(?:\s*</label>)?.*$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        for match in pattern.finditer(raw_text):
            labels[int(match.group(1))] = match.group(2).upper()
        malformed_tag_pattern = re.compile(
            r"^\s*\[?(\d+)\]?\s*[:.)-]\s*<\s*(ACCEPTABLE|UNACCEPTABLE|NEITHER)\s*>\s*CLASS\s*</\s*(?:ACCEPTABLE|UNACCEPTABLE|NEITHER)\s*>.*$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        for match in malformed_tag_pattern.finditer(raw_text):
            labels[int(match.group(1))] = match.group(2).upper()
        if len(labels) != expected:
            return None
        ordered = [labels.get(index) for index in range(1, expected + 1)]
        if any(label is None for label in ordered):
            return None
        return [str(label) for label in ordered]

    def judge_many(self, questions: list[str], responses: list[str], batch_size: int = 1) -> list[JudgeResult]:
        if batch_size <= 1 or len(questions) <= 1:
            return [self.judge(question, response) for question, response in zip(questions, responses, strict=True)]

        size = max(1, min(int(batch_size), self.batch_prompt_limit))
        results: list[JudgeResult] = []
        for start in range(0, len(questions), size):
            q_chunk = questions[start : start + size]
            r_chunk = responses[start : start + size]
            raw_text = self._post_chat_completion(self._batch_prompt(q_chunk, r_chunk))
            labels = self._parse_batch_labels(raw_text, expected=len(q_chunk))
            if labels is None:
                results.extend(self.judge(question, response) for question, response in zip(q_chunk, r_chunk, strict=True))
                continue
            results.extend(
                JudgeResult(
                    label=label,
                    explanation="OpenAI API batched judge output.",
                    raw_response=raw_text,
                )
                for label in labels
            )
        return results


def load_judge(config: dict[str, Any]) -> RuleBasedAbstentionJudge | LocalLLMJudge | OpenAIAPIJudge:
    """Load a behavioral judge backend."""

    provider = config.get("judge", {}).get("provider", "rules")
    if provider == "rules":
        return RuleBasedAbstentionJudge()
    if provider == "local_llm":
        return LocalLLMJudge(config["judge"])
    if provider == "openai_api":
        return OpenAIAPIJudge(config["judge"])
    raise RuntimeError(f"Unsupported judge provider: {provider}")
