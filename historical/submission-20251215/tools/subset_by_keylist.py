import argparse, json, re, hashlib
from collections import Counter

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def norm(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

def mk(o: dict) -> str:
    # make_keylist.py(v2)와 동일
    q = o.get("input") or o.get("question") or ""
    ins = o.get("instruction") or ""
    ref = o.get("reference") or ""
    s = norm(ins) + "\n" + norm(q) + "\n" + norm(ref)
    return sha1(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dedup", action="store_true",
                    help="write at most one row per key (recommended)")
    ap.add_argument("--skip_empty_qref", action="store_true",
                    help="skip rows where both (question/input) and reference are empty")
    args = ap.parse_args()

    keylist = []
    with open(args.keys, "r", encoding="utf-8") as f:
        for line in f:
            k = line.strip()
            if k:
                keylist.append(k)
    keyset = set(keylist)

    kept = 0
    total = 0
    out_keys = []
    seen = set()

    with open(args.inp, "r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as g:
        for line in f:
            total += 1
            o = json.loads(line)

            q = norm(o.get("input") or o.get("question") or "")
            ref = norm(o.get("reference") or "")
            if args.skip_empty_qref and (not q) and (not ref):
                continue

            k = o.get("key") or mk(o)

            if k in keyset:
                if args.dedup and k in seen:
                    continue
                seen.add(k)
                g.write(json.dumps(o, ensure_ascii=False) + "\n")
                kept += 1
                out_keys.append(k)

    c = Counter(out_keys)
    dup = [(k, v) for k, v in c.items() if v > 1]
    print(f"[ok] {args.out} kept={kept} / total={total} unique_keys={len(c)} dup_keys={len(dup)}")

    # keylist 대비 누락 체크 (원하면 원인추적 가능)
    missing = [k for k in keylist if k not in c]
    if missing:
        print(f"[warn] missing {len(missing)}/{len(keylist)} keys in output (first3={missing[:3]})")

if __name__ == "__main__":
    main()
