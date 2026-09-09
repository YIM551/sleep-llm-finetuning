import glob, json
from collections import Counter, defaultdict

paths = sorted(glob.glob("**/*pairwise*.jsonl", recursive=True))
if not paths:
    raise SystemExit("No *pairwise*.jsonl found. (검색 경로: 현재 디렉토리 이하)")

def read(path):
    for line in open(path, encoding="utf-8"):
        line=line.strip()
        if not line: 
            continue
        yield json.loads(line)

print("FOUND:")
for p in paths:
    print(" -", p)
print()

for p in paths:
    c=Counter()
    total=0
    by_match=defaultdict(Counter)
    for r in read(p):
        total += 1
        w = (r.get("winner") or r.get("choice") or r.get("result") or "").lower()
        a = r.get("model_a") or r.get("A") or r.get("left") or "A"
        b = r.get("model_b") or r.get("B") or r.get("right") or "B"

        if w in ("a","left","model_a"): key="A_win"
        elif w in ("b","right","model_b"): key="B_win"
        elif "tie" in w or w=="equal": key="tie"
        else: key="unknown"

        c[key]+=1
        by_match[f"{a} vs {b}"][key]+=1

    print(f"[{p}] total={total}  A_win={c['A_win']}  B_win={c['B_win']}  tie={c['tie']}  unknown={c['unknown']}")
    if total>0:
        winrate = c["A_win"]/total
        print(f"  -> A winrate = {winrate:.3%}")
    print()
