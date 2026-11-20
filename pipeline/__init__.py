"""TableSalt OCR model eval pipeline.

The container orchestrates: dataset -> model run -> conversion -> score -> metrics.
Model inference happens behind an HTTP provider (MLX server on the host, or Ollama),
because Docker on macOS has no Metal.
"""

__version__ = "0.1.0"
