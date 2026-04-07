# Skill: Web Research Summary

## Purpose
Search the web for information about a topic, fetch the most relevant pages,
and return a concise, well-sourced summary.

## Tier
2 (Composite — direct execution, no sub-agent loop needed)

## Tools allowed
- `web_fetch`
- `llm_call`
- `file_write`

## Steps

### 1. Decompose the query
Break the user's request into 2–4 specific search sub-questions that together
cover the topic. Write them down in your scratch notes.

### 2. Fetch sources
For each sub-question:
- Construct a search URL (use a public search engine or a known authoritative
  domain if the topic suggests one).
- Call `web_fetch` with `max_chars: 6000`.
- Record the URL and the extracted text.

Fetch at most **4 sources** total (file_budget constraint).

### 3. Synthesise
Call `llm_call` with pipeline `research` and a prompt that:
- States the original user request.
- Provides the fetched source excerpts (labelled by URL).
- Asks for a concise summary (3–6 paragraphs) with inline citations [1], [2], …
- Asks for a "Sources" section at the end listing each URL.

### 4. Write result
Call `file_write` to save the summary to `scratch.md` in your worker slot.
This is the deliverable the orchestrator will collect.

## Done looks like
- `scratch.md` exists in the worker directory.
- The file contains a summary of at least 3 paragraphs.
- At least 1 source URL is cited.
- No hallucinated facts — every claim is supported by a fetched excerpt.

## Constraints
- Do not fetch more than 4 URLs.
- Do not exceed `token_budget`.
- If a URL returns an error or empty content, skip it and note the failure.
- Keep the final summary under 800 words unless the user explicitly asked for more.
