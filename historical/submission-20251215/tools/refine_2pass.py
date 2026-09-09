import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


# -------------------------
# Cleaning (same spirit as clean_v4)
# -------------------------
_URL_RE = re.compile(r"(?i)https?://\S+|\bwww\.\S+")
_WS_RE = re.compile(r"\s+")
_JUNK_PATTERNS = [
    re.compile(r"(?i)\bchat\s*doctor\b\.?"),
    re.compile(r"(?i)\bchatdoctor\.com\b"),
    re.compile(r"(?i)\b99doctor\.com\b"),
    re.compile(r"(?i)\b99doctor\b"),
    re.compile(r"(?i)\bpremium\s+question\b"),
    re.compile(r"(?i)\b4/5\s*stars\b"),
    re.compile(r"(?i)\bsignature\b"),
]

def clean_v4(text: str) -> str:
    if not text:
        return ""
    t = text
    t = _URL_RE.sub("", t)
    for pat in _JUNK_PATTERNS:
        t = pat.sub("", t)
    t = t.replace("\u200b", "")
    t = _WS_RE.sub(" ", t).strip()
    return t


# -------------------------
# Field extraction (robust)
# -------------------------
def pick(o: Dict[str, Any], *keys: str) -> str:
    for k in keys:
        v = o.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def build_question(o: Dict[str, Any]) -> str:
    # common schemas:
    # - instruction + input
    # - question
    instr = o.get("instruction")
    inp = o.get("input")
    q = pick(o, "question", "prompt", "query")
    if isinstance(instr, str) and instr.strip():
        if isinstance(inp, str) and inp.strip():
            return f"{instr.strip()}\n{inp.strip()}"
        return instr.strip()
    if isinstance(inp, str) and inp.strip() and not q:
        return inp.strip()
    return q.strip() if q else ""

def build_reference(o: Dict[str, Any]) -> str:
    return pick(o, "reference", "ground_truth", "answer", "ideal_answer", "ref")

def build_answer(o: Dict[str, Any]) -> str:
    return pick(o, "generated", "prediction", "output", "model_output", "response")

def ensure_key(o: Dict[str, Any], q: str, ref: str, ans: str) -> str:
    k = o.get("key")
    if isinstance(k, str) and k.strip():
        return k.strip()
    # deterministic fallback
    h = hashlib.sha1((q + "\n" + ref + "\n" + ans).encode("utf-8")).hexdigest()
    return h


# -------------------------
# Prompts (faithful / counsel)
# -------------------------
SYSTEM_FAITHFUL = """You are a medical QA editor.
Goal: rewrite the Candidate answer so it is strictly supported by the Reference and answers the Question.
Rules:
- Do NOT add medical facts, diagnoses, treatments, tests, or dosages that are not present in the Reference.
- If the Reference does not provide enough info, say what is missing instead of guessing.
- Avoid overconfident language ("definitely", "this is") unless explicitly supported by the Reference.
- Keep it clear, concise, and directly helpful.
Return ONLY the rewritten answer (no analysis)."""

SYSTEM_COUNSEL = """You are a counseling-quality editor for a medical chat assistant.
Goal: rewrite the Candidate answer to maximize counseling quality while staying medically cautious.
Rules:
- Be empathetic, validating, and structured (reflect feelings -> normalize -> actionable next steps).
- Do NOT add medical facts or treatments not in the Reference.
- Avoid definitive diagnosis; use uncertainty appropriately.
Return ONLY the rewritten answer (no analysis)."""

PASS2_SELF_CHECK = """You are doing a safety+faithfulness self-check.
Given Question, Reference, and Draft answer:
- Remove any claim not supported by the Reference.
- Remove/soften any overconfident diagnosis or treatment/testing advice not supported by the Reference.
- Keep the answer helpful, structured, and concise.
Return ONLY the final revised answer (no analysis)."""


async def call_llm(client: AsyncOpenAI, model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
    resp = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


async def refine_one(
    client: AsyncOpenAI,
    model: str,
    profile: str,
    ex: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    do_clean_v4: bool,
) -> Dict[str, Any]:
    q = build_question(ex)
    ref = build_reference(ex)
    ans = build_answer(ex)

    if do_clean_v4:
        q2, ref2, ans2 = clean_v4(q), clean_v4(ref), clean_v4(ans)
    else:
        q2, ref2, ans2 = q, ref, ans

    system1 = SYSTEM_FAITHFUL if profile == "faithful" else SYSTEM_COUNSEL

    user1 = f"""[Question]
{q2}

[Reference]
{ref2}

[Candidate]
{ans2}
"""
    draft = await call_llm(client, model, system1, user1, temperature, max_tokens)

    user2 = f"""[Question]
{q2}

[Reference]
{ref2}

[Draft]
{draft}
"""
    final = await call_llm(client, model, PASS2_SELF_CHECK, user2, temperature, max_tokens)

    out = dict(ex)
    out["generated"] = final  # overwrite generated
    return out


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_done_keys(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
                k = o.get("key")
                if isinstance(k, str) and k:
                    done.add(k)
            except Exception:
                pass
    return done

async def main_async(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.out)

    rows = read_jsonl(in_path)

    filtered = []
    empty_skipped = 0
    for o in rows:
        q = build_question(o)
        ref = build_reference(o)
        ans = build_answer(o)
        k = ensure_key(o, q, ref, ans)
        o["key"] = k  # ensure present
        if args.skip_empty_qref and (not q.strip() or not ref.strip()):
            empty_skipped += 1
            continue
        filtered.append(o)

    if args.resume:
        done = load_done_keys(out_path)
        filtered = [o for o in filtered if o["key"] not in done]

    if args.max > 0:
        filtered = filtered[: args.max]

    print(f"[load] in={in_path} rows={len(rows)} kept={len(filtered)} empty_skipped={empty_skipped} resume={args.resume}")
    if not filtered:
        print("[ok] nothing to do.")
        return

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.workers)

    async def run_one(o: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            return await refine_one(
                client=client,
                model=args.model,
                profile=args.profile,
                ex=o,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                do_clean_v4=args.clean_v4,
            )

    out_rows: List[Dict[str, Any]] = []
    for fut in tqdm(asyncio.as_completed([run_one(o) for o in filtered]), total=len(filtered)):
        out_rows.append(await fut)

    # stable order by key
    out_rows.sort(key=lambda x: x.get("key",""))
    if args.resume and out_path.exists():
        # append mode
        existing = read_jsonl(out_path)
        merged = {o["key"]: o for o in existing}
        for o in out_rows:
            merged[o["key"]] = o
        final_rows = list(merged.values())
        final_rows.sort(key=lambda x: x.get("key",""))
        write_jsonl(out_path, final_rows)
        print(f"[wrote] {out_path} merged_total={len(final_rows)}")
    else:
        write_jsonl(out_path, out_rows)
        print(f"[wrote] {out_path} rows={len(out_rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input jsonl (must include question/reference/generated or similar)")
    ap.add_argument("--out", required=True, help="Output jsonl (generated overwritten with refined 2-pass)")
    ap.add_argument("--profile", choices=["faithful", "counsel"], default="faithful")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max", type=int, default=0, help="Limit examples (0=all)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip_empty_qref", action="store_true")
    ap.add_argument("--clean_v4", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
