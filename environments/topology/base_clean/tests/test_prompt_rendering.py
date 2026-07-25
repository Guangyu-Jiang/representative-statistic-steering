from __future__ import annotations

from types import SimpleNamespace

from aen_replication.models.generation import render_prompt, render_prompts


class _TokenizerWithChatTemplate:
    def __init__(self) -> None:
        self.chat_template = "stub"
        self.calls: list[dict] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        suffix = "<gen>" if add_generation_prompt else "<nog>"
        return f"{messages[-1]['content']}{suffix}"


def test_render_prompt_supports_extraction_style_chat_rendering() -> None:
    tokenizer = _TokenizerWithChatTemplate()
    bundle = SimpleNamespace(tokenizer=tokenizer)

    rendered = render_prompt(
        bundle=bundle,
        prompt_text="Who won?",
        use_chat_template=True,
        system_prompt="Answer carefully.",
        add_generation_prompt=False,
    )

    assert rendered == "Who won?<nog>"
    assert tokenizer.calls == [
        {
            "messages": [
                {"role": "system", "content": "Answer carefully."},
                {"role": "user", "content": "Who won?"},
            ],
            "tokenize": False,
            "add_generation_prompt": False,
        }
    ]


def test_render_prompts_falls_back_to_plain_text_without_chat_template() -> None:
    bundle = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=None))

    rendered = render_prompts(
        bundle=bundle,
        prompt_texts=["Q1", "Q2"],
        use_chat_template=True,
        system_prompt=None,
        add_generation_prompt=False,
    )

    assert rendered == ["Q1", "Q2"]
