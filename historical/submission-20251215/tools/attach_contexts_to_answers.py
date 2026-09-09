
import argparse, json, hashlib
from pathlib import Path
from typing import Any, Dict

def pick_question(o: Dict[str, Any]) -> str:
    for k in ["question","query","user_input","input","prompt"]:
        v=o.get(k)
        if isinstance(v,str) and v.strip():
            return v.strip()
    inst=o.get("instruction")
    inp=o.get("input")
    if isinstance(inst,str) or isinstance(inp,str):
        s=f"{inst or ''}\n{inp or ''}".strip()
        if s:
            return s
    return ""

def pick_reference(o: Dict[str, Any]) -> str:
    for k in ["reference","ground_truth","answer_gt","gt"]:
        v=o.get(k)
        if isinstance(v,str) and v.strip():
            return v.strip()
        if isinstance(v,list) and v and isinstance(v[0],str):
            return "\n".join([x for x in v if isinstance(x,str)]).strip()
    return ""

def pick_answer(o: Dict[str, Any]) -> str:
    for k in ["generated","response","answer","output"]:
        v=o.get(k)
        if isinstance(v,str) and v.strip():
            return v.strip()
    return ""

def stable_key(q: str, ref: str) -> str:
    return hashlib.sha1(f"{q}\n{ref}".encode("utf-8")).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    ctx_map={}
    with open(args.contexts,"r",encoding="utf-8") as f:
        for line in f:
            o=json.loads(line)
            k=o.get("key")
            if not k:
                continue
            ctx=o.get("retrieved_contexts") or o.get("contexts") or []
            if isinstance(ctx,list):
                ctx_map[k]=[x for x in ctx if isinstance(x,str) and x.strip()]

    out_path=Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n=0; miss=0
    with open(args.answers,"r",encoding="utf-8") as f, open(out_path,"w",encoding="utf-8") as g:
        for line in f:
            o=json.loads(line)
            q=pick_question(o)
            ref=pick_reference(o)
            ans=pick_answer(o)
            k=stable_key(q, ref)
            ctx=ctx_map.get(k)
            if ctx is None:
                miss += 1
                ctx = []
            rec={
                "key": k,
                "user_input": q,
                "response": ans,
                "reference": ref,
                "retrieved_contexts": ctx,
                # 호환용
                "question": q,
                "answer": ans,
                "ground_truth": ref,
                "contexts": ctx,
            }
            g.write(json.dumps(rec, ensure_ascii=False)+"\n")
            n += 1

    print(f"[ok] wrote {out_path} rows={n} ctx_miss={miss}")

if __name__=="__main__":
    main()
