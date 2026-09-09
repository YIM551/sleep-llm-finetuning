import argparse, json, random, re, hashlib
from collections import Counter

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def norm(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

def mk(o: dict) -> str:
    # 파일마다 question/input 키가 다를 수 있어서 안전하게 fallback
    q = o.get("input") or o.get("question") or ""
    ins = o.get("instruction") or ""
    ref = o.get("reference") or ""
    s = norm(ins) + "\n" + norm(q) + "\n" + norm(ref)
    return sha1(s)

def extract_keys(path: str):
    keys = []
    bad_empty = 0
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            o = json.loads(line)
            q = norm(o.get("input") or o.get("question") or "")
            ref = norm(o.get("reference") or "")
            # 질문/레퍼런스 둘 다 비어있으면 평가 자체가 의미 없어서 제외
            if (not q) and (not ref):
                bad_empty += 1
                continue
            k = o.get("key") or mk(o)
            keys.append(k)
    return keys, total, bad_empty

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require_unique", action="store_true",
                    help="Require key count==1 in both A and B (recommended).")
    args = ap.parse_args()

    a_keys, a_total, a_empty = extract_keys(args.a)
    b_keys, b_total, b_empty = extract_keys(args.b)

    ca = Counter(a_keys)
    cb = Counter(b_keys)

    inter = sorted(set(ca.keys()) & set(cb.keys()))

    # 충돌 대표(sha1("\n") 등) 같은 ‘빈 샘플’ 키가 끼지 않도록 2중 방어
    bad_hashes = set([
        sha1("\n"),     # adc83b...
        sha1("\n\n"),   # 71853c...
        sha1(""),       # da39a...
    ])
    inter = [k for k in inter if k not in bad_hashes]

    if args.require_unique:
        inter = [k for k in inter if ca[k] == 1 and cb[k] == 1]

    rnd = random.Random(args.seed)
    rnd.shuffle(inter)

    picked = inter[:args.n]
    with open(args.out, "w", encoding="utf-8") as g:
        for k in picked:
            g.write(k + "\n")

    print(f"[ok] wrote {args.out} n={len(picked)}")
    print(f"     A: total={a_total} kept_for_keys={len(a_keys)} empty_skipped={a_empty}")
    print(f"     B: total={b_total} kept_for_keys={len(b_keys)} empty_skipped={b_empty}")
    print(f"     intersection={len(set(ca)&set(cb))} after_filters={len(inter)} (require_unique={args.require_unique})")

if __name__ == "__main__":
    main()
