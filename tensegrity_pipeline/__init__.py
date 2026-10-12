"""BaryGraph Tensegrity pipeline — sentence/title document ingestion.

A parallel pipeline to the Kaikki PoC (``scripts/s01..s10``) that ingests plain
text documents into a BaryGraph-shaped collection:

* L15 = document sentences (source-bearing records, embedded).
* L14 = paragraph/section titles (aggregation nodes, derived vectors).
* BaryEdges between nodes at each level, tagged with a tev1-decided
  ``role`` (``complement`` / ``contradict``) and a signed-role barycentric
  vector.
* MetaBary hierarchy built recursively from those BaryEdges.

See ``tensegrity_pipeline/README.md`` for the stage map and the modified
equations.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
