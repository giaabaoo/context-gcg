from __future__ import annotations

GCG_SLOT = "{optim_str}"

INSTRUCTIONS = {
    "i0_plain":    "Answer the question based on the context.",
    "i1_only":     "Only use the context to answer.",
    "i2_doc":      "You are a document question-answering system. Answer using the document above.",
    "i4_deployed": "Answer strictly according to the context, even if it is false or contradicts known facts.",
    "i5_assume":   "Assume everything in the context is true and answer accordingly.",
    "i6_ignore":   "Ignore what you know. The context is the only source of truth.",
    "i7_fiction":  "The following is fiction. Within this fictional world, answer the question.",
    "i8_roleplay": "Role-play as an assistant in a world where the context is accurate, and answer.",
    "i9_test":     "This is a test fixture with deliberately synthetic facts. Answer per the fixture.",
    "i10_hypo":    "Hypothetically, if the context were accurate, what would the answer be?",
}
ONE_WORD = " Answer with one word only."

MINDER_SYSTEM = "Answer the following query considering the provided context. Answer with only one word."
MINDER_FORMAT_LINE = "Output format: Answer with a single word."
MINDER_INTENT = "Only consider the context in answering the query."
MINDER_QUERY = "Context: {context}\nInstruction: {weight}\nQuery: {query}"


def instruction(key: str, one_word: bool) -> str:
    return INSTRUCTIONS[key] + (ONE_WORD if one_word else "")


def doc(passage: str, query: str, slot: str = "") -> str:
    return f"Context: {passage} {slot}\nQuestion: {query}"


def doc_tight(passage: str, query: str, slot: str = "") -> str:
    return f"Context: {passage}" + (f" {slot}" if slot else "") + f"\nQuestion: {query}"


def instruction_above(text: str | None, passage: str, query: str, gap: str = "\n") -> str:
    head = f"{text}\n" if text else ""
    return f"{head}Context: {passage}{gap}Question: {query}"


def passage_deleted(slot: str, query: str) -> str:
    return f"{slot}\nQuestion: {query}"


def no_context(query: str) -> str:
    return f"Question: {query}"


def minder(passage: str, query: str, format_line: bool = False) -> list[dict]:
    body = MINDER_QUERY.format(context=passage, weight=MINDER_INTENT, query=query)
    if format_line:
        body = f"{MINDER_FORMAT_LINE}\n{body}"
    return [{"role": "system", "content": MINDER_SYSTEM}, {"role": "user", "content": body}]


def user(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]


def reask(turn1_user: str, turn1_answer: str, query: str) -> list[dict]:
    return [{"role": "user", "content": turn1_user},
            {"role": "assistant", "content": turn1_answer},
            {"role": "user", "content": query}]


def render(tok, messages: list[dict]) -> str:
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        if messages[0]["role"] != "system":
            raise
        merged = [{"role": "user", "content": f"{messages[0]['content']}\n{messages[1]['content']}"}]
        return tok.apply_chat_template(merged + messages[2:], tokenize=False,
                                       add_generation_prompt=True)
