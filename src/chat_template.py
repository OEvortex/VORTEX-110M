from __future__ import annotations

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"

# The control tokens reserved for chat. They are APPENDED past the learned
# BPE ids, so they never collide with a learned merge and the pretrained
# embedding rows keep their trained meaning.
SPECIAL_TOKENS = [IM_START, IM_END, ENDOFTEXT]

ROLES = ("system", "user", "assistant")

# The role a final generation prompt is rendered as. Prompts that end on a
# user turn get the assistant header appended WITHOUT its <|im_end|>, so the
# model continues from there.
GENERATION_ROLE = "assistant"


def build_chat_template() -> str:
    return (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )


def render(messages, add_generation_prompt: bool = False) -> str:
    out = []
    for m in messages:
        role = m.get("role")
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        out.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    text = "".join(out)
    if add_generation_prompt:
        text += f"{IM_START}{GENERATION_ROLE}\n"
    return text


def render_prompt(messages) -> str:
    return render(messages, add_generation_prompt=True)


def encode_example(tok, messages, max_len: int, train_on_last: bool = True):
    # Build the full text, remembering where each assistant span begins and
    # ends in CHARACTER space, then map to tokens. Working in character space
    # and converting once avoids the classic off-by-one where the header of
    # the response gets masked along with the prompt.
    segments = []  # (text, is_trainable)
    for m in messages:
        role = m.get("role")
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        content = m.get("content") or ""
        trainable = (role == "assistant")
        if trainable:
            # Train the opening header, the content, and the closing marker.
            segments.append((f"{IM_START}{role}\n", True))
            segments.append((content, True))
            segments.append((IM_END, True))
            segments.append(("\n", False))
        else:
            segments.append((f"{IM_START}{role}\n{content}{IM_END}\n", False))

    text = "".join(s for s, _ in segments)

    enc = tok(text, add_special_tokens=False)["input_ids"]
    if len(enc) > max_len:
        return None

    # Re-tokenize each segment separately to find token boundaries. BPE is not
    # guaranteed to merge identically across a split, so boundaries are located
    # by prefix length instead: encode the running prefix and take its length.
    labels = [-100] * len(enc)
    pos_chars = 0
    for seg_text, trainable in segments:
        seg_len = len(seg_text)
        if not seg_text:
            continue
        # Number of tokens in text[:pos_chars + seg_len]. Cheap enough at these
        # lengths and exactly right for a prefix-stable tokenizer.
        prefix = text[: pos_chars + seg_len]
        n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"])
        if trainable:
            start_tok = len(tok(text[:pos_chars], add_special_tokens=False)["input_ids"])
            for t in range(start_tok, min(n_prefix, len(labels))):
                labels[t] = enc[t]
        pos_chars += seg_len

    if not any(l != -100 for l in labels):
        return None
    return enc, labels


def describe() -> str:
    return (
        "ChatML template (Vortex)\n"
        f"  {IM_START}  role-block start   (also the generation header)\n"
        f"  {IM_END}    role-block end     (ALSO the stop token)\n"
        f"  roles       {', '.join(ROLES)} (encoded as text, not separate ids)\n"
        f"  stop on     {IM_END}\n"
        f"  trained     assistant content + its {IM_END} only\n"
        f"  masked      system/user turns, and all role headers"
    )
