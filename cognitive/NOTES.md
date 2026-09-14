# arXiv text search — known issue (deferred until s04 done)

## Symptom
arxiv_search with multi-word queries returns mostly stem-bleed garbage
(e.g. "CO2 curing carbonation cement" -> carbon-materials papers).

## Root cause
cognitive_arxiv.text_search index was built with MongoDB default english
stemmer. "carbonation" stems to "carbon", so it matches thousands of
irrelevant "carbon" papers. Probe: top 200 hits for "carbonation" contain
the literal word in only 2 documents. Single-term queries whose word == its
own stem (cement, 458 docs) work fine.

$language:'none' at QUERY time does NOT fix it: document tokens were stemmed
at INDEX build time, so a none-language query cannot match stemmed tokens
(literal "concrete" => 0 docs vs 24k under english).

## Fix (after s04)
1. Drop cognitive_arxiv.text_search index.
2. Recreate with {"default_language": "none"}.
3. In scripts/mcp_int_cog.py _arxiv_search_body, pass
   {"$search": query, "$language": "none"} to the $text filter.

Or in shell:
  db.cognitive_arxiv.dropIndex("text_search")
  db.cognitive_arxiv.createIndex({title:"text", abstract:"text"}, {default_language:"none", name:"text_search"})

## Note on corpus coverage (not a bug)
arXiv genuinely has ~0 papers using literal "carbonation"/cement-carbonation
vocabulary; that domain lives in journals (Cement and Concrete Research,
etc.), not arXiv. So even after the fix, a cement-carbonation search will
return few hits — the fix only removes the false "carbon" matches.
