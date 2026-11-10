"""pdf2md: PDF -> page images -> local vision model -> Markdown.

Each page is sent as a separate request (no context bleed between pages) and
cached to <output>_pages/page-NNNN.md, so an interrupted run resumes where it
stopped. The final Markdown file is reassembled from the cache after every page.
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from importlib.resources import files
from pathlib import Path

import pymupdf
from openai import APIConnectionError, APIStatusError, OpenAI

DEFAULT_BASE_URL = "http://localhost:1234/v1"
USER_MESSAGE = "Convert this page."


def load_prompt(path: Path | None) -> str:
    if path:
        return path.read_text(encoding="utf-8")
    return files("pdf_to_mrkdwn").joinpath("prompt.txt").read_text(encoding="utf-8")


def parse_pages(spec: str | None, total: int) -> list[int]:
    """'1-3,7,10-' -> 1-based page numbers, clamped to the document."""
    if not spec:
        return list(range(1, total + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a) if a else 1
            end = int(b) if b else total
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    bad = [p for p in pages if p < 1 or p > total]
    if bad:
        sys.exit(f"error: page(s) {sorted(bad)} out of range 1-{total}")
    return sorted(pages)


def pick_model(client: OpenAI, requested: str | None) -> str:
    try:
        ids = [m.id for m in client.models.list().data]
    except APIConnectionError:
        sys.exit(
            f"error: can't reach the server at {client.base_url}\n"
            "  Is LM Studio's server running? (Developer tab -> Start Server)"
        )
    if requested:
        if requested not in ids:
            print(f"warning: '{requested}' not in server's model list: {ids}", file=sys.stderr)
        return requested
    matches = [i for i in ids if "wevisdoc" in i.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(ids) == 1:
        return ids[0]
    sys.exit(
        "error: couldn't pick a model automatically; pass --model with one of:\n  "
        + "\n  ".join(ids or ["(server lists no models)"])
    )


def render_page(doc: pymupdf.Document, page_no: int, dpi: int) -> bytes:
    return doc[page_no - 1].get_pixmap(dpi=dpi).tobytes("png")


def convert_page(client: OpenAI, model: str, system: str, png: bytes,
                 max_tokens: int, temperature: float) -> tuple[str, str | None]:
    b64 = base64.b64encode(png).decode()
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": USER_MESSAGE},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
    )
    choice = r.choices[0]
    return (choice.message.content or "").strip(), choice.finish_reason


def assemble(cache_dir: Path, out_path: Path, page_markers: bool) -> int:
    parts = []
    for f in sorted(cache_dir.glob("page-*.md")):
        text = f.read_text(encoding="utf-8").strip()
        if page_markers:
            n = int(f.stem.split("-")[1])
            text = f"<!-- page {n} -->\n{text}"
        parts.append(text)
    out_path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return len(parts)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="pdf2md",
        description="Convert a PDF to Markdown page-by-page with a local vision model (LM Studio).",
    )
    ap.add_argument("pdf", type=Path, help="input PDF")
    ap.add_argument("-o", "--output", type=Path, help="output .md (default: <pdf>.md next to the PDF)")
    ap.add_argument("-p", "--pages", help="pages to convert, e.g. '1-3,7,10-' (default: all)")
    ap.add_argument("--model", help="model id on the server (default: auto-detect 'wevisdoc')")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"server URL (default: {DEFAULT_BASE_URL})")
    ap.add_argument("--dpi", type=int, default=200, help="render resolution (default: 200)")
    ap.add_argument("--max-tokens", type=int, default=6000, help="max output tokens per page (default: 6000)")
    ap.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (default: 0)")
    ap.add_argument("--timeout", type=float, default=600, help="per-request timeout, seconds (default: 600)")
    ap.add_argument("--prompt", type=Path, help="custom system prompt file (default: WeVisDoc's training prompt)")
    ap.add_argument("--force", action="store_true", help="re-convert pages that are already cached")
    ap.add_argument("--keep-images", action="store_true", help="also save rendered page PNGs in the cache dir")
    ap.add_argument("--no-page-markers", action="store_true", help="omit <!-- page N --> markers")
    args = ap.parse_args(argv)

    if not args.pdf.is_file():
        sys.exit(f"error: {args.pdf} not found")
    out_path = args.output or args.pdf.with_suffix(".md")
    cache_dir = out_path.with_name(out_path.stem + "_pages")
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key="lm-studio", timeout=args.timeout, max_retries=1)
    model = pick_model(client, args.model)
    system = load_prompt(args.prompt)

    doc = pymupdf.open(args.pdf)
    pages = parse_pages(args.pages, doc.page_count)
    print(f"{args.pdf.name}: {len(pages)} of {doc.page_count} pages -> {out_path}  (model: {model}, {args.dpi} dpi)")

    truncated, failed = [], []
    t_all = time.monotonic()
    for i, n in enumerate(pages, 1):
        page_file = cache_dir / f"page-{n:04d}.md"
        if page_file.exists() and not args.force:
            print(f"[{i}/{len(pages)}] page {n}: cached, skipping")
            continue
        png = render_page(doc, n, args.dpi)
        if args.keep_images:
            (cache_dir / f"page-{n:04d}.png").write_bytes(png)
        t0 = time.monotonic()
        try:
            text, finish = convert_page(client, model, system, png, args.max_tokens, args.temperature)
        except APIStatusError as e:
            msg = str(e)
            hint = ""
            if "context" in msg.lower() or "n_ctx" in msg.lower():
                hint = "  -> raise the model's context length in LM Studio (16384+) or lower --dpi"
            print(f"[{i}/{len(pages)}] page {n}: FAILED ({e.status_code}) {msg[:200]}{hint}", file=sys.stderr)
            failed.append(n)
            continue
        except APIConnectionError as e:
            print(f"[{i}/{len(pages)}] page {n}: FAILED (connection/timeout) {e}", file=sys.stderr)
            failed.append(n)
            continue
        page_file.write_text(text + "\n", encoding="utf-8")
        assemble(cache_dir, out_path, not args.no_page_markers)
        note = ""
        if finish == "length":
            truncated.append(n)
            note = "  WARNING: hit --max-tokens, output truncated"
        print(f"[{i}/{len(pages)}] page {n}: {len(text)} chars in {time.monotonic() - t0:.1f}s{note}")

    count = assemble(cache_dir, out_path, not args.no_page_markers)
    print(f"\ndone in {time.monotonic() - t_all:.0f}s: {count} page(s) in {out_path}")
    if truncated:
        print(f"truncated pages (re-run with higher --max-tokens and --force -p ...): {truncated}")
    if failed:
        print(f"failed pages (just re-run the same command to retry them): {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
