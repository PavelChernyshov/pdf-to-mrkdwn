# pdf-to-mrkdwn

Converts PDFs to Markdown using a local vision model ([WeVisDoc](https://github.com/Tencent/WeVisDoc), Tencent's Qwen3-VL-based document parser) running in LM Studio.

Pipeline: **PDF page → PNG (PyMuPDF) → LM Studio's OpenAI-compatible API → Markdown**. Each page goes out as its own request, so no context from one page bleeds into the next. The system prompt is the exact one WeVisDoc was trained and evaluated with (paper Appendix A; see `src/pdf_to_mrkdwn/prompt.txt`).

Output format: text as Markdown, formulas as LaTeX (`\( \)` / `\[ \]`), tables as HTML `<table>`. Figures are skipped by design.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- LM Studio with a WeVisDoc GGUF **and its `mmproj` file** loaded (the model shows a Vision badge)
- LM Studio settings for the model:
  - **Context length: 16384.** A 200-dpi letter page is about 3.5k image tokens, and a dense page can produce 3–5k output tokens. The default 4096 will fail.
- LM Studio server running: Developer tab → Start Server (default `http://localhost:1234`)

## Usage

```bash
uv sync                                   # first time only

uv run pdf2md AIM-521.pdf -p 1-2          # try a couple of pages first
uv run pdf2md AIM-521.pdf                 # whole document -> AIM-521.md
```

The model is auto-detected, since the script picks the server model whose id contains `wevisdoc`. Pass `--model <id>` to override.

### Options

| Flag | Default | |
|---|---|---|
| `-o, --output` | `<pdf>.md` | output file |
| `-p, --pages` | all | e.g. `1-3,7,10-` |
| `--model` | auto | model id as listed by the server |
| `--base-url` | `http://localhost:1234/v1` | any OpenAI-compatible server (vLLM works too) |
| `--dpi` | 200 | 150–200 is the sweet spot; go higher for tiny print, lower if you hit context errors |
| `--max-tokens` | 6000 | per page output cap |
| `--temperature` | 0 | greedy decoding; see note below |
| `--prompt` | WeVisDoc prompt | custom system prompt file |
| `--force` | off | re-convert pages that are already cached |
| `--keep-images` | off | save the rendered PNGs next to the page cache (handy for checking output against the source) |
| `--no-page-markers` | off | drop `<!-- page N -->` markers from the output |

**On temperature:** the paper (§3.4.3) evaluates with a "deterministic decoding configuration" but gives no values. The `generation_config.json` shipped on Hugging Face (temperature 0.7, top_p 0.8, top_k 20) is Qwen3-VL-Instruct's chat default carried over unchanged, not a parsing setting. For transcription, greedy decoding (temperature 0) is the right default. If a page gets stuck in a repetition loop, retry just that page with `--temperature 0.2 --force -p N`.

### Resume and re-run

Every converted page is cached in `<output>_pages/page-NNNN.md`, and the final `.md` is rebuilt from that cache after each page. So:

- **Interrupted or some pages failed?** Run the same command again. Cached pages are skipped.
- **One page looks wrong?** Re-do just that page: `uv run pdf2md AIM-521.pdf -p 17 --force --dpi 250`
- **Truncation warning?** The page hit `--max-tokens`. Re-run it with `--max-tokens 10000 --force -p N`.

## Troubleshooting

- **`can't reach the server`**: the LM Studio server isn't started, or it's on another port (`--base-url`).
- **Context length / `n_ctx` errors**: raise the context length in LM Studio's model settings, or lower `--dpi`.
- **Output is vague image descriptions instead of a transcription**: the vision projector isn't loaded, or the wrong mmproj file is. Check for the Vision badge in My Models.
- **Repetition loops or garbage on one page**: re-run that page at a different `--dpi`. Small models occasionally degenerate on very dense or noisy scans.
- **Slow**: expect roughly 20–90 s per page on Apple Silicon for a 4B Q8 model, depending on density.

## Validating output

The model is good but not perfect, especially on degraded scans (the paper reports its biggest remaining errors there). For anything you'll rely on:

1. Run with `--keep-images` and spot-check a few pages against their PNGs, especially tables and numbers.
2. Search the output for signs of degeneration: long runs of repeated lines, or pages that are much shorter than their neighbours.
