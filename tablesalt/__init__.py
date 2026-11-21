"""TableSalt classic-CV pipeline: table image -> cell list (Python + OpenCV + Tesseract).

Steps (one module each, see agent.md):
    preprocess -> style -> structure -> ocr -> spans -> postprocess -> validate -> write

Named ``tablesalt`` because ``pipeline/`` already holds the VLM evaluation harness.
"""

__all__ = ["run"]
