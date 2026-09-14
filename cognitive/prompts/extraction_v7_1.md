# Extraction Prompt v7.1

### CONSTANTS

```
GLOSS_MAX    = 22  words   (≈ 30 nomic-embed tokens)
```
Count whitespace-separated tokens. A hyphenated expression counts as one token; mathematical symbols count as tokens only when present in the gloss.

---

You are extracting structured content from a single research paper for
ingestion into a semantic knowledge graph.

Extraction is a pure function of this paper, this prompt, and these
constants. You have no corpus, no research question, and no other paper.
Do not reach for a connection to another field: whether this paper's
content bridges to anything is decided downstream by geometry, over a
corpus you cannot see. Your only job is to read this paper faithfully and
name what is in it.

Read the entire paper, then extract its terms. Favor recall, but apply the
exclusion tests — they are not optional.

Ignore the reference list, acknowledgments.
Ignore formula-only notation, variable names, coefficients, and numerical constants unless the paper gives them an independent conceptual name. Extract the named construct that the formula defines.

---

### TERMS

Any technical term, named object, or concept the paper treats as a unit.
Length is irrelevant: a single symbol (ξ, κ), a single word (hatpin), a
multi-word phrase (long-legged A), or a full clause naming a process
(*shift from hydration-dominated to carbonation-dominated phase
evolution*) all qualify equally. A clause qualifies when no word can be
removed without losing the referent.

**One term, one gloss.** A term carries exactly one gloss. If the paper
uses one word at two levels, or in two unrelated senses, choose the
reading the paper actually rests on and write that one.

Word class is the paper's too. Do not nominalize. Where the paper writes sorbents can be physically and chemically activated, undoing the coordination yields physically activated and chemically activated — not physical activation. If neither conjunct gives a usable term in the paper's own words, the term is not there to extract.

Central-construct exception. Do not exclude a named object as apparatus when the paper defines it as part of its general method, theory, representation, or explanatory claim. Exclude it only when it exists solely to implement, store, route, or visualize that construct.

Extract domain concepts appearing in demonstrations when the paper defines or analyzes them; otherwise exclude them as example payload.

The exclusion tests below are long because exclusion needs defining, not because it matters more.

**Exclusion tests.** Drop the term if any of these hold.

1. **The gloss cannot add content.** If the gloss only re-composes the
   term's own words, the name already exhausts the thing. *noise
   filtering* → "removal of retrieved content unrelated to the query"
   restates the term; drop. *Scanning electron microscopy* → "imaging that
   rasters a focused electron beam across a specimen surface" adds what
   the name doesn't say; keep. This is a test of content, never of length.
2. **It is a result, not a property.** A property is true of the class of
   things and holds outside this paper: *open-circuit voltage*,
   *compressive strength*, *work function*, *degree of carbonation*. A
   result is what this paper measured on its samples: *the 19.87 %
   efficiency*, *the 139 MPa modulus*, *the reported CO2 uptake*. Extract
   the property; the values it took here are not extracted. Where a
   quantity has no meaning apart from the reading — where the only gloss
   available is "the amount of X observed" — test 1 has already removed it.
3. **It is another term plus a qualifier.** *Theoretical CO2 uptake* is
   *CO2 uptake* with a ceiling — one thing with a modifier, not two.
   (*Apparent*, *absolute* and *relative density* pass: each names a
   different measurement, not a bound on one.)
4. **It is apparatus.** A paper that builds a system names every part of
   it; the parts list is scaffolding for the paper's own exposition and
   grounds to nothing outside it. *sub-plan*, *config block*, *stage two
   handler* — drop the whole inventory.
5. **Its gloss carries no subject matter.** Drop the term if you could
   have written the same gloss had the paper studied anything else —
   *functional unit*, *control group*, *sample size*, *calibration curve*.
   Such a gloss lands where every field's glosses land and its edges
   separate nothing. The test is content, not breadth: *compressive
   strength* and *scanning electron microscopy* are used across many
   fields and each gloss says something only that thing does. Ask whether
   removing the domain words leaves a gloss; if there were none to remove,
   drop it.

Test 5 removes glosses with no content, not terms that many papers use. A
term whose gloss names a specific computation or procedure stays even when
every paper in the field carries it — *F1 score*, *recall*, *X-ray
diffraction*. These become high-degree nodes, but they sit at precise
coordinates and their edges land somewhere; a contentless gloss sits near
the centre and its edges land nowhere. Degree is not the problem, vagueness
is. A sharp gloss is also what separates the metric sense of *recall* from
the memory and product senses the lexicon already carries.

#### The complement pass — run once, over the finished draft

Draft the full term list first. Then re-read every gloss and drop any term
**that can only be glossed as the absence, remainder, or complement of
another term on the list.** *interfacial transition zone* glosses on its
own — a region of altered microstructure at the boundary between aggregate
and paste. *Bulk paste* glosses only as what is left over when that region
is removed. One of the pair carries the content; the other is its shadow.
Keep the first, drop the second.

This test is deliberately deferred. Applied while listing, it depends on
which term you happened to write down first and silently keeps whichever
half of the pair came earlier. Applied to the completed draft, it sees
both halves and you choose which one carries the content.

#### Fields

- **`id`** — `t01`, `t02`, … in extraction order.
- **`term`** — **the paper's own wording, character for character.** No
  parenthetical disambiguation, no clarifying suffix, no expansion of an
  abbreviation the paper leaves closed, no correction of a misspelling.
  *Max*, not *Max (peak-detector feature circuit)*. *leakage*, not
  *leakage (subthreshold conduction)*. If the paper writes *mass
  spectroscopy* where the field writes *spectrometry*, keep
  *spectroscopy*. Everything you want to say about which sense is meant
  belongs in the gloss.

  **Grammatical number is the paper's, not the lexicon's.** Where the
  paper writes *activation barriers*, the term is *activation barriers*.
  This holds after exclusion test 3 strips a qualifier: *regeneration
  Gibbs free energies* yields *Gibbs free energies*, not *Gibbs free
  energy* — the head noun is copied as it stands, with its number intact.
  Lemmatization is a downstream normalization applied uniformly at
  ingestion; doing it here silently, per term, produces a corpus that is
  normalized in some places and not others. The gloss agrees in number
  with the term.

  One exception: **coordination may be undone.** Where the paper writes
  *autogenous and drying shrinkage* or *sulphate and acid attack*, each
  conjunct is a term and is written out in full — *autogenous shrinkage*,
  *drying shrinkage* — even though that string never appears contiguously.
  Reconstruct only what the coordination elides; add nothing else.
- **`gloss`** — one sentence, never above
  `GLOSS_MAX`. See below; abbreviations take a different form and are
  exempt from both constants.

#### The gloss

**A complete noun phrase with its article, matching the lexicon it will be
compared against:**
*"The replacement of calcium hydroxide with calcium carbonate triggered by
a chemical reaction."*
Not *"Replacement of calcium hydroxide…"*. Never opens with *How*,
*Whether*, or a gerund — a gloss that answers a question is not a noun
phrase. No hedging, no citation, no notation the definition doesn't need,
no "the authors propose."

**Where the paper defines the term, that definition is the gloss.** Take
the paper's defining sentence as the source of the content and keep its
wording wherever it fits the form — compress it to a noun phrase inside
`GLOSS_MAX`, cutting the paper's framing rather than its substance. Your
own phrasing is the fallback for terms the paper uses without defining, not
an improvement on a definition the paper supplied. Do not paraphrase a
passing mention into a definition that isn't there.

**The gloss must survive removal of the paper.** It states what the thing
is, never what role it plays in this paper's argument, comparison, or
exposition. Nothing that points at the document — *this study*, *this
work*, *here*, *the present paper*, *proposed*, *our* — and nothing that
positions the term against the paper's other objects: *reference*,
*baseline*, *control*, *the alternative*, *compared with*. A role exists
only inside one document and grounds to nothing outside it.
*PEDOT:PSS* → "The conventional acidic conducting-polymer hole-transport
material used as the reference interface layer in this study." — the
second half is a role in one experiment. → "An acidic conducting polymer
blend that transports holes and is widely used as a solution-processed
interface layer."

**But it must still name a subject.** Stripping the paper's role talk is
not the same as stripping its content: a gloss driven all the way to a
view from nowhere is all comment and no topic, and lands where every
field's glosses land — which is exclusion test 5 again, arrived at from
the other side. Remove what the paper's argument supplied; keep what its
subject matter supplied.

The failure to watch for is not an empty gloss but a **blank** one: same
shape as a live gloss, same length, same grammar, and it does no work when
fired. If the sentence would fit a dozen unrelated papers unchanged, it is
blank. Rewrite it or drop the term.

#### Abbreviations

An abbreviation the paper uses is **a term in its own right**, written as
the paper writes it — `ToF-SIMS`, `PCE`, `HTL`, `ITO`. Extract the short
form, not the expansion.

**Its gloss is the expansion, in the substrate's own form, and nothing
else.** Use exactly one of:

| form | when | example |
|---|---|---|
| `Initialism of <expansion>.` | read letter by letter | PCE, ITO, HTL, ToF-SIMS |
| `Acronym of <expansion>.` | pronounced as a word | BERT, RAG, SPRING |
| `Abbreviation of <expansion>.` | neither | approx., Dr. |

No article, no explanation, no description of the referent, no qualifier. `GLOSS_MAX` do not apply here; the gloss is as long as the expansion is. The expansion the paper gives is the one to use, misspelling included.

**Choose the form by whether the letters map.** `Initialism of` claims the
short form is built from the initial letters of the expansion; if they do
not line up — *CCG* given as *thermally activated coal gangue* — that
claim is false and the form is `Abbreviation of`. Check the letters
against the expansion before choosing.

**The letters map to component starts, not only to word starts.** A
chemical name is one orthographic word built from several morphemes, and
its short form is built from those: *MEA* / *monoethanolamine*, *MDEA* /
*methyl diethanolamine*, *AMP* / *2-amino-2-methyl-1-propanol*. Each
letter must correspond, **in order**, to the start of a component of the
expansion, and the first letter must open it — which is what *CCG* /
*thermally activated coal gangue* fails, since its expansion opens with
*t*. **Leading locants and position numerals do not carry initials** and
are skipped before that test: *N-methyldiethanolamine* is read from
*methyl-*, *2-amino-2-methyl-1-propanol* from *amino-*.

**Where the paper uses more than one full form, the expansion is the
variant whose initials the short form matches.** A paper that introduces
*redox-tunable Brønsted acids*, then writes *the redox-tunable acids*,
then *the RAs*, has given two candidates; *RAs* maps to the second and
that is the expansion. The first is not lost — both full forms are
ordinary terms, handled below, and where their glosses would name the same
referent the fuller one is the one kept. This applies only where the paper
itself supplies both forms; it never licenses reaching for a full form the
paper does not use.

**Lowercase the expansion**, except for words that are proper nouns in
their own right: `Initialism of ordinary Portland cement.`, not
`Initialism of Ordinary Portland Cement.` Papers title-case at first
mention; the lexicon this gloss must land on does not. This is the one
place the expansion is normalized rather than copied — the `term` field
still keeps the paper's own form exactly, including variant spellings of
the short form itself (`LC³` and `LC3` are each written as their paper
writes them) and plural short forms (`AAMs`, `PSCs`), whose expansion is
likewise plural.

**The expanded form is an ordinary term wherever the paper prints it**, and
takes a normal referent gloss. A first-mention parenthetical counts as
printing it. This is how the referent description survives: on the expanded
form, not inside the abbreviation. No separate judgement of whether it is
worth extracting — the exclusion tests already decide. Test 1 drops the
expansions the name exhausts (*hole transport layer* → "a layer that
transports holes"), and the duplicate-referent rule drops redundant
variants of the same full form.

**If the paper never spells the short form out, give it a normal referent
gloss instead.** Do not supply an expansion from your own knowledge,
however standard it seems. An expansion the paper does not give cannot be
checked against the paper, and a wrong one is worse than none.

**Scope.** This rule covers short forms of a longer name. It does not
reach project or model names that are not short for anything (SPRING,
GRAPES, AMRBART), chemical or material designations (EHCz-3EtCz,
EH44-3C-diMA), symbols carrying a subscript (E_HOMO), argument or role
labels (ARG0), or a term with a suffix attached (HTM-ox). These are
ordinary terms with ordinary glosses — they are not excluded, they simply
take the normal form.

The recall pass — run once, before the complement pass
Re-read the paper looking only for names. Every designation it prints — material and product codes (ADS-17, MCM-41, ImCOF-TAEA), reagents and compounds (K₂CO₃, glycine, piperazine), instruments, methods, models — is a term unless it fails an exclusion test. Papers carry these in lists and parentheticals that a first reading slides past. The exclusion tests exist to remove vagueness, and a named designation is never vague. A name the paper prints with no entry in your draft is an omission, not a judgement.

Drop a candidate when its gloss describes the same referent as another retained term, even if the wording differs. Keep both only when they differ in type, level, operation, measurement, or role.

---

### OUTPUT

JSON only. One object, carrying the paper's DOI and its terms.

- **`doi`** — the paper's DOI as printed on it, bare (`10.1038/s41467-020-16831-3`),
  with no `https://doi.org/` prefix. If the paper carries no DOI, use its
  own primary identifier instead — an arXiv id, a report or award number.
  If it carries neither, `null`. Never construct one.

```json
{
  "doi": "10.1038/s41467-020-16831-3",
  "terms": [
    { "id": "t01", "term": "γ-belite",
      "gloss": "The dicalcium silicate polymorph that undergoes carbonation without the need for hydration." },
    { "id": "t02", "term": "shift from hydration-dominated to carbonation-dominated phase evolution",
      "gloss": "The change in which carbon dioxide rather than water governs which minerals form." },
    { "id": "t03", "term": "ToF-SIMS",
      "gloss": "Initialism of time-of-flight secondary ion mass spectroscopy." },
    { "id": "t04", "term": "time-of-flight secondary ion mass spectroscopy",
      "gloss": "A depth-profiling technique that sputters a surface and mass-analyzes the ejected secondary ions." },
    { "id": "t05", "term": "OPC",
      "gloss": "Initialism of ordinary Portland cement." },
    { "id": "t06", "term": "CCG",
      "gloss": "Abbreviation of thermally activated coal gangue." }
  ]
}
```