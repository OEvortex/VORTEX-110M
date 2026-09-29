"""
ChatML-style chat template for the Vortex models.

Why ChatML
----------
The tokenizer is a plain 16K byte-level BPE with NO reserved control tokens, so
a chat format has to be designed rather than inherited. ChatML is the right
shape for it for one specific reason: it needs only FOUR new ids, and all four
can be carved out of existing vocabulary without retraining the tokenizer.

    <|im_start|>   start of a role block
    <|im_end|>     end of a role block, and end of generation
    <|endoftext|>  document separator (already exists as EOS)

Roles are encoded as TEXT inside the block ("system", "user", "assistant")
rather than as separate tokens. That is deliberate:

  * 4 new ids instead of ~10 (`<|system|>`, `<|user|>`, ... would each cost an
    id, and a 16K vocab has no ids to spare -- every one is a parameter).
  * The role words are already high-frequency English tokens, so they are
    nearly free at the BPE level.
  * The model can still generalize to an unseen role string, which cannot
    happen with a closed set of role tokens.

Generation stops on `<|im_end|>`.

Sequencing
-----------
A single rendered conversation looks like:

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What is 2+2?<|im_end|>
    <|im_start|>assistant
    It is 4.<|im_end|>

Note there is NO trailing newline after the final `<|im_end|>`. Generation
appends `<|im_end|>` and stops; anything after it is not part of the target.

Loss masking
------------
Only ASSISTANT content is trained on. The tokens of the prompt -- the
`<|im_start|>` marker, the role word, and the whole user/system turn -- are
labelled `-100`. The assistant's opening `<|im_start|>assistant` header IS
trained, because the model has to learn to open its own turn; the closing
`<|im_end|>` is trained so it learns to stop.

Mismatched template and masking is the single most common SFT bug: train the
prompt too and the model learns to echo questions before answering, and never
learns to emit `<|im_end|>` so it rambles until the context fills.
"""
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
    """The Jinja chat template, for `tokenizer.apply_chat_template`."""
    return (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )


def render(messages, add_generation_prompt: bool = False) -> str:
    """Render a message list to the ChatML training string.

    `messages` is [{"role": ..., "content": ...}, ...]. Raises on an unknown
    role rather than silently training the model on a malformed turn.
    """
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
    """Render up to (but not including) the assistant's reply.

    The trailing `<|im_start|>assistant\n` IS included, because that is what the
    model conditions on when it starts generating.
    """
    return render(messages, add_generation_prompt=True)


def encode_example(tok, messages, max_len: int, train_on_last: bool = True):
    """Tokenize one conversation into (input_ids, labels) with masking.

    Returns None when the conversation does not fit or contains no trainable
    assistant turn -- both are normal and should be filtered, not crashed on.

    Masking rule, matching the docstring: every system/user turn, and the
    `<|im_start|>role` headers, are -100. An assistant turn's content and its
    closing `<|im_end|>` are trained.
    """
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
    """Human-readable summary, printed at the top of an SFT run."""
    return (
        "ChatML template (Vortex)\n"
        f"  {IM_START}  role-block start   (also the generation header)\n"
        f"  {IM_END}    role-block end     (ALSO the stop token)\n"
        f"  roles       {', '.join(ROLES)} (encoded as text, not separate ids)\n"
        f"  stop on     {IM_END}\n"
        f"  trained     assistant content + its {IM_END} only\n"
        f"  masked      system/user turns, and all role headers"
    )
