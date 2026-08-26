def format_sources(hits: list[dict]) -> str:
    lines = []
    seen: set[tuple] = set()
    for hit in hits:
        key = (hit.get("filename"), hit.get("page"), hit.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        page = hit.get("page")
        page_bit = f", p.{page}" if page not in (None, "") else ""
        chunk_bit = f" (chunk {hit.get('chunk_id')})" if hit.get("chunk_id") is not None else ""
        lines.append(f"- {hit.get('filename', 'unknown')}{page_bit}{chunk_bit}")
    return "\n".join(lines)


def append_citations(answer: str, hits: list[dict]) -> str:
    sources = format_sources(hits)
    if not sources:
        return answer
    if "**Sources**" in answer or "\nSources" in answer:
        return answer
    return f"{answer.rstrip()}\n\n**Sources**\n{sources}"


def hits_to_tool_text(hits: list[dict]) -> str:
    if not hits:
        return "No relevant passages found."
    blocks = []
    for i, hit in enumerate(hits, start=1):
        page = hit.get("page")
        page_bit = f", page {page}" if page not in (None, "") else ""
        blocks.append(
            f"[{i}] {hit.get('filename', 'unknown')}{page_bit} | chunk {hit.get('chunk_id')}\n"
            f"{hit.get('text', '').strip()}"
        )
    return "\n\n".join(blocks)
