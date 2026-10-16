"""Split at headings, then at the embedding tokenizer's token budget."""

import hashlib
import re

from repopilot.rag_models import Chunk, Document


def chunk_documents(documents: list[Document], tokenizer, max_tokens=180) -> list[Chunk]:
    chunks = []
    for document in documents:
        # Markdown headings and common reStructuredText underline headings.
        sections = re.split(r"(?m)(?=^#{1,6} |^[^\n]+\n[=~-]{3,}\s*$)", document.text)
        index = 0
        for section in sections:
            if not section.strip():
                continue
            # Offsets cut the original text, retaining punctuation/code verbatim.
            offsets = tokenizer(section, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
            for start in range(0, len(offsets), max_tokens):
                group = offsets[start:start + max_tokens]
                text = section[group[0][0]:group[-1][1]].strip()
                if not text:
                    continue
                chunk_id = hashlib.sha256(f"{document.document_id}:{index}:{text}".encode()).hexdigest()[:32]
                chunks.append(Chunk(**document.model_dump(exclude={"text"}), text=text, chunk_id=chunk_id, chunk_index=index))
                index += 1
    return chunks
