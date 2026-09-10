# Stage notes — accepted findings (not yet fixed/addressed)

## Stage timing ledger — barygraph_all build (measured) and POC reference

Measured from logs/state files (times UTC). Use these instead of rough guesses for
future ETA estimates. "--all" = barygraph_all (kaikki.org-dictionary-all), "--poc" =
barygraph_poc (kaikki-en, ~1.06M L15 pairs). Stage order guard: each stage must
wait for the previous; measured stages only.

| stage | start | end | wall | units processed | rate |
|---|---|---|---|---|---|
| s01 parse (--all) | 08-29 00:42:49 | 01:13:28 | 0h 31m | 10,757,861 lines → 10.3M words / 12.3M senses | ~5,850 lines/s |
| s02 embed (--all) 1st attempt | 08-29 01:13:47 | ~08-31 23:41 (crash ENOSPC) | ~70.5h | 7,743,488 embedded | ~31 embeds/s |
| s02 embed (--all) resume | 09-01 03:09:21 | 09-02 21:53:16 | 42.7h | remainder 4,981,570 (total 12,725,058) | ~32 embeds/s |
| s03 insert (--all, v4) | 09-03 19:59:25 | 09-04 05:07:53 | 9h 9m | 23,207,138 docs | ~705 docs/s |
| s04 L15 load/stream (--all) | 09-04 13:26:54 | 14:51:26 | 1h 25m | 12,718,626 senses into mmap | ~2,500/s |
| s04 HNSW build 4096→1024 (--all) | 14:51:26 | 15:18:50 | 27m 24s | 12.7M vectors (ef_c=100, M=16) | ~7.7–12.8k/s |
| s04 greedy match (--all) | ~15:18:50 | 16:18:25 | ~1h 0m | 5,332,670 pairs | — |
| s04 embed+insert loop (--all) | 16:19:47 | 09-08 04:14:12 | 3d 11h 54m | 10,416 batches × 512 = 5,332,670 BEs | ~17.6 edges/s (~34s/batch) |
| s04 stamp bulk (crash) | 04:14:12 | (timeout) | — | ~6.0M of ~10.7M stamps applied | crash site :330 |
| s04 total to crash | 09-04 13:26:50 | 09-08 04:14:12 | 3d 14h 47m | | |
| s04 (--poc) whole stage | 08-18 09:17:32 | 10:17:49 | 1h 0m | 1,039,813 pairs + 14,416 orphan re-entry | NOT embed-representative |
| s05 word vecs (--poc, scoped) | 08-23 10:40:24 | 10:41:44 | 80s | 6,711 L14 word vectors (no embed) | — |
| s06 L14 edges (--poc) | 08-23 10:42:44 | 10:57:07 | 14m 23s | 1,472,061 words paired, 6 tiers | — |
| s07 L14 orphan re-entry (--poc) | 08-23 11:07:13 | 11:18:18 | 11m 05s | 5,144 orphans vs 1,423,669 BEs | — |
| s08 MetaBary (--poc) | 08-23 11:25:30 | 11:40:12 | 14m 42s | 7,299 triads, stops at L9 (0 triads) | — |

Throughput rules of thumb for future ETAs:
- s02 embed ≈ 30/s; s04 type_text embed ≈ 15–18/s (≈34s per 512 batch) — these two
  are the pipeline's long poles; s04 total ≈ 5.33M edges at ~17.6/s ≈ 3.5 days.
- s04 rank-1 estimate check: initial guess 2–3 d; corrected to 3.5–4 d from the
  15–17/s measurement; actual 3d 14h 47m — the corrected estimate was right.
- s04 orientation: sense stream (1.4h) + HNSW (0.5h) + greedy (1h) ≈ 3h fixed
  overhead before the embed loop begins; the ~3.5-day loop is the predictor.
- s03 ≈ 9h at ~705 docs/s; s06–s08 (--poc) were ~11–15 min each but scale with
  the L14 word/BE counts, not with the s04 sense count.
- Resume overhead: --force re-streams all 12.7M senses into a fresh 208GB mmap
  (~1h at 2.5k/s, degraded bursts to ~105s/100k ≈ 3h logged) — see the
  mmap-rebuild note below; a reuse-eager resume would skip most of this.

## s03 / s04 — kaikki duplicate entries are kept, not deduped
Exact-duplicate sense entries (same word, pos, lang, gloss from separate kaikki
entries) remain separate nodes; s04 pairs them as L15 "merge" edges (q=1.0),
consuming 2 sense nodes each that could otherwise pair cross-word.
- Measured: ~19.2% of top-of-stream L15 edges are same-lang same-gloss dups;
  ~9.7% same-word cross-lang (legit homograph links, not dups); 0 polysemous
  same-word merges (polysemy_floor working).
- All L15 edges are untyped (edge_type=None); typed only at L14 (s06).
- Acceptance: keep for now. Follow-up: dedupe (word,pos,lang,gloss) at s03
  ingest, merging etymology/alternates → free ~19% pairing capacity. Requires
  re-run of affected stages; not applicable mid-s04.

## s04 — effective embed rate ~15-17/s (below the ~28/s measured)
L15 type_text batch ~34s per 512 → ETA ~3.5-4 days (not 2-3). Accepted.
Follow-up: investigate batch-size ceiling / type_text+nb() overhead.
Measured: main embed+insert loop ran at ~17.6 edges/s, total 3d 14h 47m to
the crash (see timing ledger above) — the revised ETA bracket held.

## s04 — checkpoint JSON only saved at finish()
cp.processed updates in memory per batch; the file stays processed:0 until
finish(). Live progress must be read via Mongo count of
{doc_type:baryedge, level:15}. Accepted; follow-up: periodic flush / metrics
feed (Babylon Bridge).

## s04 — deferred parent-stamp bulk_write is a crash site (NetworkTimeout)
Main pairing accumulates every sense→BE parent_edge_id UpdateOne (~10.7M = 2 ×
5.3M) into ONE ordered=False bulk_write at the very end of the loop
(scripts/s04_l15_edges.py:330). At full scale the single call exceeded mongod's
120s socket timeout → NetworkTimeout, killed run() before orphan re-entry (4e)
ran and before the checkpoint was written. Partial application: ~6.0M of the
~10.7M stamps landed (5,332,670 BEs themselves were already safely inserted per
batch). Recovery: relaunch with --force — it skips main pairing, treats stamped
senses as paired, and reaches orphan re-entry, which stamps per-orphan, so the
crash site (line 330) is bypassed. Status 2026-09-08: the client now defaults to
600s via PYMONGO_SOCKET_TIMEOUT_MS for s04 (see mmap note); the megabatch still
needs a real fix — flush stamp writes incrementally inside the loop (~50-100k/
batch) so a tail timeout can't strand the stage. NOT YET DONE.

## s04 — --force resume now reuses the V mmap (FIXED 2026-09-08)
The load phase used to open S04_MMAP_PATH with mode="w+" unconditionally,
re-streaming all 12.7M L15 sense vectors from Mongo (~1-3h) even when --force
skipped main pairing — and its O_TRUNC destroyed the existing 208GB file that
already held the identical dataset (exact shape match).
Fix (scripts/s04_l15_edges.py): on --force with an existing file of the exact
expected byte size, open mode="r+", tail-zero-scan for the last valid row, and
backfill only the missing tail by _id (measured: 183 rows out of 12,718,626).
Additionally feeds cursors through PYMONGO_SOCKET_TIMEOUT_MS (lib/db.py env
override, default 120s) — the long-resume streams now use 600s so a busy mongod
getMore can't kill the stage (this was crash #2's cause). Remaining TODO: the
s04 deferred parent-stamp bulk_write (see next note) still uses the same client
and could exceed the 600s window at full scale — flush incrementally.

## s04 — orphan re-entry accelerated (A+B+D, 2026-09-08)
Naïve 4e over the full build would take ~13 d: ~8 d for the exact 4096-d
argmax over 5.33M BEs (OV@BEV.T ≈ 2.9e17 flops, measured 417 GFLOP/s) + ~4.5 d
for 6.7M per-orphan embeds. Fixed (scripts/s04_l15_edges.py):
- B) parent selection via projected-space hnswlib ANN (MATCH_DIM=1024, reuse
  lib.match.ann_index/project_match), refined as exact argmax among the top-K
  (S04_ORPHAN_ANN_K=32) candidate BEs in projected space → ~1 h. Approximate,
  consistent with the accepted main-loop ANN-in-projected-space design.
- A) orphan type_text only uses (word,pos,lang) + synonyms/antonyms (never the
  sense gloss) ⇒ all senses of a word are byte-identical ⇒ embed once per
  distinct (word,pos,lang), not per orphan (~1-1.5M embeds ≈ ~1 d). Zero
  behavior change vs the old loop.
- D) ids/words (~5 GB) persisted to /storage/bary/s04_ids_words.pkl after the
  load; a crashed resume now restores them instead of re-streaming 12.7M
  senses (~3.5 h). Sidecar removed by _cleanup_mmap on stage completion.
Memory at 4e: be_vecs 87 GB (Julia-style: kept, needed full-dim for bary_vec)
+ BEVp 22 GB + HNSW ~23 GB; NOT the 87+110 GB (BEV stack + OV copy) the old
code would have held.
Revised 4e estimate: ~1.2-1.5 d (embed ~1 d + ANN ~1 h + insert/bulk ~4-8 h) ↘
vs ~13 d.

## Global — 4096-dim vector bloat (256 MB/sense file, 87 GB BE list)
Everything holds float32 4096-dim vectors: V mmap 208 GB on disk (12.7M rows),
BE vectors 87 GB in RAM during s04 4e. For a PoC this is acceptable at current
scale but is the single biggest resource line item. Follow-up (post-build):
project to MATCH_DIM once and persist projected (n²→ 1024-d), or quantize
(float16/uint8). Not urgent; revisit after s09.

## s04 — stream EOF blowup fixed via ObjectId cutover bound (2026-09-08)
The sense stream is `find(node/sense/15) + sort(_id,1)` — a full _id-index
walk over the ENTIRE collection. That collection grew to 28.3M docs once s04
inserted 5.33M BEs, whose ObjectIds sort AFTER every s03 word/sense. To prove
EOF the final getMore had to fetch+filter the whole non-matching BE tail —
5.3M+ docs in one batch — blowing even a 600 s socketTimeoutMS. This killed
force v1 (120s) and force5 (600s), both at ~12.5-12.7M of 12.7M.
Fix: cap the scan at `$lte ObjectId("2026-09-04T12:00:00Z")`. Justification embeds
in code: ObjectId timestamps are immutable, s03 (ending 09-04T05:07:53Z) is the
ONLY writer of sense docs, and every later pass (s04 BE/orphan re-entry etc.)
inserts >= 09-08. Scan region is back to words+senses only; row order unchanged
so V-mmap reuse stays deterministic. Also raised client socket timeout for s04
to 1800 s (PYMONGO_SOCKET_TIMEOUT_MS) as belt-and-braces.
(Attempted alternatives discarded: reverse max-sense-_id walk = an 18-min
fetch-filter of the same tail; a covering {doc_type,node_type,level,_id} index =
index build wedging mongod for 10+ min at 5%; both reverted/dropped.)

## lib/db.py — global socketTimeoutMS=120000
Long reads (e.g. full-collection $sample over 12.7M) still time out. The s04
crash was fixed via covering index {doc_type,node_type,level} + cap fallback
(count 12.7M in ~4s). Avoid heavy ad-hoc aggregations during pipeline runs.

## lib/match.py — ANN match uses Gaussian 4096->1024 projection
Reduces HNSW index ~210GB->55GB so pairing fits RAM; ranking cosine measured
in reduced space (approximate, equivalent recall on synthetic). Accepted
design decision.

## Global — pre-existing test failures (unrelated to pipeline)
test_sense_node_schema + test_assoc_search fail since float32 storage commit
fca4f80 (tests not updated). Out of scope.

## s04 — 430GB embed-cache hostage (FIXED 2026-09-08)
`EMBED_CACHE_FILE=data/parsed_all/embed_cache.jsonl` is a 430GB JSON append-log.
`CachedEmbedder.__init__` (lib/embed.py:46-60) loads the WHOLE file into a
key→vector dict — ~105GB anonymous RSS + a 66-min silent startup window (and
only ~6.4M of ~26M lines parsed before a truncated line aborts the load).
This was also the dominant co-cause of the force7 OOM (cache 105 + be_vecs 87 +
BEVp 22 + hnsw 22 ≈ 236GB on a 251GiB host).
Fix: launch s04 with `EMBED_CACHE_FILE=""` in the env — lib/config.py:111-113
maps empty → None (cache disabled), no code change. verify before any stage
that falls back to per-batch Ollama embedding.

## s04 — overnight crash ledger + supervision (2026-09-08→09)
- gen1 (force v9, PID 102741): died 00:16:08 UTC on Ollama embed transport
  timeout — 4 retries (backoff 5/10/20s) exhausted at s04:273 via
  lib/embed.py:136. NOT an OOM. Ollama reachability must be confirmed before
  any Phase-A run.
- gen2 (112284): SIGKILLed 04:35:32 by the supervisor's OOM guard at RSS
  202,730,088 kB (~193GB) — the ANN-phase ramp below.
- gen3 (121086): relaunched 04:41:32, killed manually on top of the fix below.
- Supervision now: /tmp/opencode/super_s04.sh — logs RSS every 60s to
  /tmp/opencode/s04_mem.log, auto-relaunch with ≥300s crash-loop backoff,
  SIGKILL guard when VmRSS > 200,000,000 kB (~190.7GiB), exits when
  pipeline_state_all/04_l15_edges.json shows done:true. A smaps tripwire
  (/tmp/opencode/smaps_capture.sh, thresholds 120/150/180GiB) dumps top VMA
  regions so a recurrence pinpoints the allocating object. Units bug in v1
  (kB vs MB) fixed 06:55 — do not repeat (compare GiB = kb/1048576).

## s04 — ANN-phase RSS root cause: V.mmap file-backed page-in (CONFIRMED 2026-09-09)
Longstanding question "why does RSS ramp ~150GB during the 42-min ANN parent
search?" — answered by the smaps capture at RSS 121GiB:
- 72GB `rw-s` (file-backed memmap) = /storage/bary/s04_V.mmap. The sweep
  reads all 6.7M orphan sense rows through the 208GB memmap (~110GB of pages),
  which land in page cache and are COUNTED IN RSS; the kernel never evicts them
  because the process barely allocates during the scan.
- Remaining RSS: BEVp 20GB + hnswlib's own copy 21GB + base ≈ 178-185GB peak.
  This exactly reproduces gen1's curve (27 → 175-182GB, −29GB after del/gc).
- The BEVp[lab] einsum was NOT the culprit. The column-wise refine (CHUNK_Q
  65k→16k) is good hygiene but does NOT stop the file-backed ramp.
- REAL FIX (DONE 09-09, running as gen6 PID 146527): per-chunk
  POSIX_FADV_DONTNEED on the served row slice (os.posix_fadvise on the V file
  fd, coalesced into contiguous runs) — drops the touched file-backed pages
  back to disk immediately after the chunk's refine, in the ANN sweep AND in
  each insert-batch. Also applied in the gen5+ restart path. NOT YET DONE as
  of the gen4 death; implemented hours later.
- gen4 DEATH (09-09 08:40:11Z): supervisor OOM_GUARD SIGKILL at RSS
  200,739,008 kB (~191.4GiB) — 86% through the ANN sweep (5.78M/6.7M), ~10 min
  from finishing. File-backed page-in beat the predicted 185GB peak; the guard
  (200,000,000 kB) worked as designed. gen5 (145925, relaunched 08:46) was
  killed 08:52 to load the fix as gen6 instead of dying again.
- Measured: ANN parent search = 6,718,626 orphans × CHUNK_Q=1024-dim projected,
  k=32, ≈ 42 min (~2,700/s) at peak ~185GB — expected to finish under the
  190.7GiB guard and roll into Phase A (embed) where the pages idle harmlessly.
- gen4 (PID 125743) ran the column-wise variant: RSS 121GiB @ 43%, 181GB @ 77%
  — tracking the file-backed model exactly; peak projects ~185GB.

## Downstream scale audit — s05–s09 sized for the 768-d PILOT (2026-09-09)
Whole suite pre-allocates float32 matrices at settings.embed_dim (4096 for
qwen3-embedding:8b) and full-corpus counts (10,256,975 L14 words, 5.33M L15
BEs). Every stage will OOM/crash as written. Matching is already safe
(lib.match projects to MATCH_DIM=1024; top_k_pairs/ann_index/greedy scale);
the landmines are the full-dim materializations:
- BOOTSTRAP check: properties.word+pos+lang index EXISTS → s05's per-word
  sense queries are covered; s05 itself is fine (streams, batches).
- s06 (L14 edges): `V = np.empty((n_words, 4096))` = ~168GB RESIDENT → must be
  np.memmap on /storage (s06_V.mmap, delete after); ALSO per-tier edge_docs are
  unbounded (each doc holds a 4096-d list ≈ 135KB) → must stream inserts in
  batch_n slices like s07's write loop. FIX PENDING.
- s07 (L14 orphan re-entry): prealloc OV capped at 2,000,000 orphans →
  IndexError at ~8M; brute-force `OV[start:end] @ BEV.T` ≈ 12 CPU days at full
  scale (2.9e17 FLOPs). Must reuse the s04 machinery verbatim: memmap OV →
  project → HNSW over projected BEV → knn_query + column-wise refine +
  MADV_DONTNEED per chunk; fetch parent docs (incl. 4096-d vectors) per insert
  batch, never the whole BE pool (~100GB otherwise). FIX PENDING.
- s08/s09 (MetaBary): `_load_unparented_bes` builds (n, 4096) np.empty in RAM —
  87GB for 5.33M unparented L15 BEs → memmap; write phase pages in only the
  winning rows. top_k_pairs + bridge HNSW already project → fine. FIX PENDING.
- Pending env/time knobs: batch_size default 512; ANN_THRESHOLD 20k; MATCH_DIM
  1024; embed_dim 4096 (authoritative from EMBED_DIM).
## 2026-09-09 — gen6 ANN DONE, cold-start timeout crash, gen7 auto-relaunch
- gen6 (146527): ANN sweep finished 10:52:41Z (0:37:15) — the fadvise fix is MOOT
  (kernel never drops mmap'd pages via POSIX_FADV_DONTNEED) BUT the anon-based
  guard carried it through 190GB file-cache RSS untouched. Sweep itself proven
  safe; V page-in is reclaimable and now irrelevant.
- gen6 died 11:01Z on the 4th embed transport timeout: Ollama cold-loads the
  model after the idle ANN stretch (keep_alive default 5m); every attempt raced
  a ~60-120s cold load + 512-text batch, dying at ~120s each. EXACT gen1 mode.
- Root fix: lib/embed.py:104 cap 180→600s; .env.build-all
  EMBED_TIMEOUT_SECONDS=900 (supervisor's next launch sources it). gen7 gets
  min(default 600, cap 600)=600s — cold-start amortizes inside ONE attempt now.
- Supervisor automation worked end-to-end: detected death ~60s, relaunched
  gen7 (166431) 11:02:27Z, --force resume via s04_V.mmap (12718626 rows valid).
- Warmup call earlier set qwen keep_alive=2h; 600s windows make re-cold-load a
  non-event.
- gen7 embed ETA ~15:00 UTC+2. After that, ~1-day Phase A embed + insert loop.
