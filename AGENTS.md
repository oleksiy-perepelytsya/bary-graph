# Project Goal (persist across sessions)

The long-term aim of this whole effort — beyond any single feature, benchmark, or
batch run — is to **improve the agent's cognitive memory** and, ultimately, to give
the agent a **personality-like layer built on BaryGraph Structure MetaBarys (SMBs)**.

Concretely:
- The cognitive-pipeline work (extraction, SMB analysis, proposals, batches,
  `cog_*` MCP tools, `opencode run` cycles with an ollama model) is a means to
  that end, not an end in itself.
- Every working piece of the pipeline — terms in `cognitive/batches/terms_batch.jsonl`,
  proposals in `smb_proposals.jsonl`, validated SMB triads — is material that
  should eventually feed the model's own persistent associative memory and
  persona layer.
- Work should therefore favor the *real pipeline* and its durable artifacts
  (`cog_cycle.py`, `cog_loop2.py`, `mcp_int_cog.py`, batch JSONL files) over
  throwaway shims. If a piece of plumbing is being tested, test it the way it
  will run in production (GPU will be available and fast; CPU slowness is a
  transient constraint, not a design driver).
- No SMB coercion: proposals are coordinates to investigate, and only the user
  decides when proposals become real SMBs in Mongo.