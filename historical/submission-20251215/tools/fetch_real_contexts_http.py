
import argparse, json, hashlib, time
from pathlib import Path
from typing import Any, Dict, List

import requests

def pick_question(o: Dict[str, Any]) -> str:
    for k in ["question", "query", "user_input", "input", "prompt"]:
        v = o.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    inst = o.get("instruction")
    inp = o.get("input")
    if isinstance(inst, str) or isinstance(inp, str):
        s = f"{inst or ''}\n{inp or ''}".strip()
        if s:
            return s
    return ""

def pick_reference(o: Dict[str, Any]) -> str:
    for k in ["reference", "ground_truth", "answer_gt", "gt"]:
        v = o.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v and isinstance(v[0], str):
            return "\n".join([x for x in v if isinstance(x, str)]).strip()
    return ""

def stable_key(q: str, ref: str) -> str:
    return hashlib.sha1(f"{q}\n{ref}".encode("utf-8")).hexdigest()

def extract_contexts(resp_json: Any) -> List[str]:
    if isinstance(resp_json, dict):
        # 1) direct list[str]
        for k in ["retrieved_contexts", "retrievedContexts", "contexts", "context", "chunks"]:
            v = resp_json.get(k)
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                return [x.strip() for x in v if x and x.strip()]

        # 2) list[dict] -> content/text/...
        for k in ["documents", "docs", "results", "items", "top_k"]:
            v = resp_json.get(k)
            if isinstance(v, list) and v:
                out=[]
                for it in v:
                    if isinstance(it, str) and it.strip():
                        out.append(it.strip()); continue
                    if isinstance(it, dict):
                        for kk in ["content","text","page_content","chunk","passage"]:
                            vv = it.get(kk)
                            if isinstance(vv, str) and vv.strip():
                                out.append(vv.strip()); break
                if out:
                    return out

        # 3) nested
        for k in ["retrieval", "rag", "data"]:
            v = resp_json.get(k)
            if isinstance(v, dict):
                got = extract_contexts(v)
                if got:
                    return got
    return []

def load_done_keys(out_path: Path) -> set:
    done=set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                o=json.loads(line)
                k=o.get("key")
                if k: done.add(k)
            except Exception:
                pass
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--topk", type=int, default=18)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry_sleep", type=float, default=1.0)
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = load_done_keys(out_path) if args.resume else set()

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            o=json.loads(line)
            q=pick_question(o)
            ref=pick_reference(o)
            if not q and not ref:
                continue
            k=stable_key(q, ref)
            rows.append((k,q,ref))
    if args.max > 0:
        rows = rows[:args.max]

    mode = "a" if (args.resume and out_path.exists()) else "w"
    kept=0
    with out_path.open(mode, encoding="utf-8") as g:
        for (k,q,ref) in rows:
            if k in done_keys:
                continue

            body = {"query": q, "topK": args.topk, "namespace": None}

            last_err=None
            for t in range(args.retries):
                try:
                    r = requests.post(args.endpoint, json=body, timeout=args.timeout)
                    r.raise_for_status()
                    j = r.json()
                    ctx = extract_contexts(j)

                    # dedup (순서 유지)
                    seen=set()
                    ctx2=[]
                    for x in ctx:
                        xx=x.strip()
                        if not xx or xx in seen:
                            continue
                        seen.add(xx)
                        ctx2.append(xx)

                    rec = {
                        "key": k,
                        "user_input": q,
                        "reference": ref,
                        "retrieved_contexts": ctx2,
                        # 호환용
                        "question": q,
                        "ground_truth": ref,
                        "contexts": ctx2,
                    }
                    g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                    last_err=None
                    break
                except Exception as e:
                    last_err=e
                    time.sleep(args.retry_sleep)

            if last_err is not None:
                raise SystemExit(f"[ERR] endpoint call failed after retries. last={last_err}")

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"[ok] wrote {out_path} rows={kept}")

if __name__ == "__main__":
    main()
