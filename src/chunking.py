"""Structure-aware chunking: one pass, heading stack, tables row-wise, images captioned."""
import io
import uuid
import base64
from langfuse import observe
from .config import vision_client, VISION_MODEL
from .parsing import get_item_text, classify, get_page


class Section:
    def __init__(self, title: str, level: int, parent=None):
        self.title = title
        self.level = level
        self.parent = parent

    def path(self):
        titles, node = [], self
        while node is not None and node.parent is not None:
            titles.append(node.title)
            node = node.parent
        return list(reversed(titles))


@observe(name="caption_image")
def caption_image(item, doc, section_context: str):
    """Caption with Groq vision (fixed: original used undefined vision_model.generate_content)."""
    try:
        pil_image = item.get_image(doc)
        if pil_image is None:
            return None
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        resp = vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in 1-3 sentences. Focus on data, labels, "
                                f"numbers, or visible text. Section: '{section_context}'."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Skipping an image under '{section_context}': {e}")
        return None


@observe(name="build_document_index")
def build_document_index(doc, user_id: str, doc_id: str, doc_name: str):
    root = Section("Document Root", level=-1)
    heading_stack = [root]
    chunks = []
    element_counter = 0

    for item, level in doc.iterate_items():
        category = classify(item)
        page = get_page(item)

        if category == "heading":
            heading_text = get_item_text(item)
            while len(heading_stack) > 1 and heading_stack[-1].level >= level:
                heading_stack.pop()
            node = Section(heading_text, level, parent=heading_stack[-1])
            heading_stack.append(node)
            if len(heading_text) >= 3:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "element_id": element_counter,
                    "text": heading_text,
                    "metadata": {
                        "type": "heading",
                        "section": " > ".join(node.path()) or "Document Root",
                        "page": page if page is not None else 0,
                        "table_id": -1,
                    },
                })
            element_counter += 1
            continue

        current_section = " > ".join(heading_stack[-1].path()) or "Document Root"

        if category == "table":
            table_id = element_counter
            try:
                df = item.export_to_dataframe(doc=doc)
                for row_idx, row in df.iterrows():
                    row_text = f"Table (Section: {current_section}): " + " | ".join(
                        f"{col}: {val}" for col, val in row.items()
                    )
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "element_id": element_counter,
                        "text": row_text,
                        "metadata": {
                            "type": "table_row",
                            "section": current_section,
                            "page": page if page is not None else 0,
                            "table_id": table_id,
                            "row_index": int(row_idx),
                        },
                    })
            except Exception as e:
                print(f"Skipping a table under '{current_section}': {e}")
            element_counter += 1
            continue

        if category == "picture":
            caption = caption_image(item, doc, current_section)
            if caption:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "element_id": element_counter,
                    "text": f"Image (Section: {current_section}): {caption}",
                    "metadata": {
                        "type": "image_caption",
                        "section": current_section,
                        "page": page if page is not None else 0,
                        "table_id": -1,
                    },
                })
            element_counter += 1
            continue

        text = get_item_text(item)
        if len(text) < 5:
            element_counter += 1
            continue

        chunks.append({
            "id": str(uuid.uuid4()),
            "element_id": element_counter,
            "text": text,
            "metadata": {
                "type": category,
                "section": current_section,
                "page": page if page is not None else 0,
                "table_id": -1,
            },
        })
        element_counter += 1

    for c in chunks:
        c["metadata"]["user_id"] = user_id
        c["metadata"]["doc_id"] = doc_id
        c["metadata"]["doc_name"] = doc_name
    return chunks
