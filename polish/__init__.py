"""Team 14 'polish' modules (from elmallah111-design/Lerrai-team-14-, branch preprocess).

enhance.enhance(img)        classic image enhancement: auto-diagnosis, shadow, deskew,
                            perspective, despeckle, optional grid-line removal, upscale
repair.repair_cells(cells)  learned post-processing: header/label vocabulary snapping and
                            per-table number-format rebuild (vocab in learned/vocab.json,
                            learned from generated training tables, never from data/dev)
No model weights, CPU only, milliseconds per table.
"""
