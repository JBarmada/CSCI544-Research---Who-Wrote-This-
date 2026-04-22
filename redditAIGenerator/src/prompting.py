from src import config


PROMPT_VERSION = "reddit_ai_v1"


def prompt_template_id(mode: str) -> str:
    return f"{PROMPT_VERSION}_{mode}"


def build_messages(mode: str, source_text: str) -> list[dict[str, str]]:
    base_system = (
        "You write fluent Reddit-style text. "
        "Do not mention prompts, instructions, or that you are an AI. "
        "Return only the requested text."
    )

    if mode == "controlled_rewrite":
        user = (
            "Rewrite the Reddit post below so it stays on the same topic, keeps a "
            "similar level of detail, and reads like a fresh standalone Reddit post "
            "or comment written by another user.\n\n"
            f"Original text:\n{source_text}"
        )
    elif mode == "continuation":
        user = (
            "Continue the Reddit text below with a plausible next section. "
            "Keep the same topic and tone, but write new content that extends the "
            "post naturally.\n\n"
            f"Seed text:\n{source_text}"
        )
    elif mode == "style_conditioned":
        user = (
            "Rewrite the Reddit text below into a more polished, structured, and "
            "persuasive Reddit-style response while keeping the same core topic. "
            "Do not mention that it is rewritten.\n\n"
            f"Original text:\n{source_text}"
        )
    else:
        raise ValueError(f"Unsupported generation mode: {mode}")

    return [
        {"role": "system", "content": base_system},
        {"role": "user", "content": user},
    ]


def build_all_mode_prompts(source_text: str) -> dict[str, list[dict[str, str]]]:
    return {mode: build_messages(mode, source_text) for mode in config.GENERATION_MODES}
