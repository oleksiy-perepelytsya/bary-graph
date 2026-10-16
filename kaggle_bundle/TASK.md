# Test: extract a case matrix from US court opinions

## What is in this archive
- `cases/` holds 100 published US court opinions, taken from CourtListener (bulk data of 2026-03-31).
  - Each opinion is one file, `<opinion_id>.txt`, containing the full opinion text.
- `cases.csv` lists the opinions: opinion_id, court, court level, case name, date, number of characters, sha256 of the text, and the CourtListener URL.
- `matrix_template.json` is the matrix structure, with the allowed values for each field.

The sample covers these court levels:

| Court level | Opinions |
|---|---|
| U.S. Supreme Court | 1 |
| Federal appellate | 34 |
| Federal district | 20 |
| State supreme | 25 |
| State appellate | 20 |

All opinions were decided from 1990 on, are 6,000–60,000 characters long, and are published.

## The task
For each opinion, fill in one matrix: a JSON object with the fields of `matrix_template.json`.

**Output:** one file per opinion, `out/<opinion_id>.json`, or a single `out.jsonl` with one object per line.

## Fields

| Field | What to fill in |
|---|---|
| `case_id` | The opinion_id, taken from the file name. |
| `case_name` | Name of the case. |
| `court` | Name of the court. |
| `court_level` | One value from the template. |
| `jurisdiction` | `federal`, or the name of the state. |
| `date_decided` | Date of the decision, YYYY-MM-DD. |
| `docket_number` | Docket number. |
| `procedure_type` | `appeal` / `certiorari` / `original_action` / `motion_ruling` / `trial_judgment` / `habeas` / `petition_for_review` / `other`. |
| `area_of_law` | One value from the template. |
| `judges` | Judges with their role: `author`, `panel`, `concurring` or `dissenting`. |
| `parties` | Who is the appellant/petitioner and who is the appellee/respondent. Also note the parties' roles in the lower court. |
| `lower_court` | The lower court and its ruling, if this is an appeal. |
| `question_presented` | The main question before the court, in one sentence. |
| `facts` | List of the key facts. |
| `arguments` | Arguments of the parties: `[{party, argument, succeeded, evidence_quote, legal_basis}]`. **`evidence_quote` must be a verbatim substring of the opinion text, at least 12 characters, unchanged.** |
| `legal_issues` | Each legal issue, the court's answer to it, and the side it was decided for. |
| `standard_of_review` | Standard of review: `de_novo` / `abuse_of_discretion` / `clear_error` / `substantial_evidence` / `plain_error` / `other` / `none`. |
| `reasoning` | The court's key reasoning, as a list. |
| `holding` | The decisive holding, in 1–2 sentences. |
| `citations` | Cited cases: name, citation (e.g. `550 U.S. 544`) and treatment (`followed` / `distinguished` / `overruled` / `criticized` / `cited`). |
| `statutes_applied` | Statutes, rules and constitutional provisions applied, e.g. `42 U.S.C. § 1983`, `Fed. R. Civ. P. 12(b)(6)`. |
| `disposition` | Outcome: `affirmed` / `reversed` / `vacated` / `remanded` / `affirmed_in_part_reversed_in_part` / `dismissed` / `granted` / `denied` / `other`. |
| `winning_party` | `petitioner_or_appellant` / `respondent_or_appellee` / `mixed`, or `""` if there is no winner. |
| `separate_opinions` | Concurrences and dissents: type, author, and the gist in one sentence. |
| `costs_fees` | Who pays the costs and fees, and the amount in USD, if stated. |
| `decisive_factors` | Decisive factors, as a list. |

## Rules
1. Everything must come from the text of the opinion. Nothing invented. Unknown means empty (`""`, `[]`, `null`).
2. Each `evidence_quote` is a literal piece of the text. We check this automatically by substring match against `<opinion_id>.txt`. An argument without such a quote does not count.
3. Process the full text, without truncation.
4. `disposition` and `winning_party` must not contradict each other.
5. Use only the values from the closed lists in the template.

## How we check
We check:
- that all fields are present and the values are valid;
- verbatim quotes (the share of `evidence_quote` values found in the text);
- agreement with the CourtListener metadata (court, date, case name);
- a manual check of a sample: `disposition`, `winning_party`, `holding` and the completeness of the arguments, citations and statutes;
- the absence of invented content.

Please also report the method, the model, the time taken and the cost per opinion.
