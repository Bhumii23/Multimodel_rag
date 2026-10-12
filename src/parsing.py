"""PDF parsing with Docling. No chunking here, just convert PDF -> document."""
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langfuse import observe

_pipeline = PdfPipelineOptions()
_pipeline.images_scale = 2.0
_pipeline.generate_page_images = True
_pipeline.generate_picture_images = True
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline)}
)


@observe(name="parse_pdf")
def parse_pdf(pdf_path: str):
    result = converter.convert(pdf_path)
    print(f"Parsed: {pdf_path}")
    return result.document


def get_item_text(item) -> str:
    # Docling: .text in most versions, .get_text() in others. Never str(item) (it's repr).
    for attr in ("text", "caption_text"):
        v = getattr(item, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    try:
        t = item.get_text()
        if isinstance(t, str) and t.strip():
            return t.strip()
    except Exception:
        pass
    s = str(item).strip()
    if s.startswith(type(item).__name__):
        return ""
    return s


def classify(item) -> str:
    try:
        label = str(getattr(item, "label", "")).lower()
    except Exception:
        label = ""
    raw = type(item).__name__
    if "SectionHeader" in raw or "Title" in raw or "section_header" in label or label.endswith("title"):
        return "heading"
    if "Table" in raw or label == "table":
        return "table"
    if "Picture" in raw or label == "picture":
        return "picture"
    if "List" in raw or "list_item" in label:
        return "list_item"
    if "Text" in raw or "paragraph" in label or "text" in label or "code" in label or "formula" in label:
        return "text"
    return "text" if get_item_text(item) else "other"


def get_page(item):
    # Docling exposes page via item.prov[0].page_no, NOT item.page_number
    try:
        return item.prov[0].page_no if item.prov else None
    except Exception:
        return None
