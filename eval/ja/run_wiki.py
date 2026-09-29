"""Wiki judgments with Laya on a real markdown wiki (frontmatter + relative .md links).

Ground truth is derived mechanically from the wiki itself:
  link     : existing [text](./x.md) links = positives; unlinked targets = negatives (easy: random, hard: same tag)
  sense    : two adjacent paragraphs of one page = same sense; paragraphs from two sibling pages = different
  type     : frontmatter `type` predicted from title+summary (knowledge concept/synthesis/preference/thinking)
Usage (server running): uv run python eval/ja/run_wiki.py /path/to/wiki/notes [--n 100]
Only aggregate numbers are written (results_wiki.json); no wiki text is stored.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from run import result_name, MLX, Client, summarize

FM = re.compile(r"\A---\n(.*?)\n---\n", re.S)
LINK = re.compile(r"\[([^\]]+)\]\((\./[^)#\s]+\.md)(?:#[^)]*)?\)")


def load(root: Path, sub: str):
    pages = {}
    for p in sorted((root / sub).glob("*.md")):
        t = p.read_text(encoding="utf-8")
        m = FM.match(t)
        if not m:
            continue
        fm = {}
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip().strip('"')
        tags = re.findall(r"[\w\-]+", fm.get("tags", ""))
        pages[p.name] = {"type": fm.get("type", ""), "title": fm.get("title", p.stem),
                         "summary": fm.get("summary", ""), "tags": set(tags), "body": t[m.end():]}
    return pages


def paragraphs(body):
    out = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if 60 <= len(para) <= 500 and not para.startswith(("|", "```", "#", "-", "*", ">")):
            out.append(para)
    return out


def noul(c, state, q):
    resp, _, err = c.ask(MLX, state, {"q": {"type": "noul", "instructions": q}})
    if resp is None:
        return None
    p = resp["answers"]["q"]["noul"]
    return p


def run_binary(c, cases, q, truth_key="gold"):
    """cases: list of (state, gold_bool, tag). Returns metrics + threshold sweep."""
    ps = []
    for state, gold, tag in cases:
        p = noul(c, state, q)
        if p is not None:
            ps.append((p, gold, tag))
    pairs = [(max(p, 1 - p), (p >= .5) == g) for p, g, _ in ps]
    res = summarize(pairs, len(cases) - len(ps))
    # best threshold + AUC-like separation
    pos = [p for p, g, _ in ps if g]
    neg = [p for p, g, _ in ps if not g]
    res["mean_p_pos"] = round(sum(pos) / max(1, len(pos)), 3)
    res["mean_p_neg"] = round(sum(neg) / max(1, len(neg)), 3)
    auc = sum((a > b) + .5 * (a == b) for a in pos for b in neg) / max(1, len(pos) * len(neg))
    res["auc"] = round(auc, 3)
    best = max(((sum((p >= t) == g for p, g, _ in ps) / len(ps), t) for t in [i / 20 for i in range(1, 20)]))
    res["best_thr_acc"] = [round(best[0], 3), best[1]]
    for tag in sorted({t for *_, t in ps}):
        sub = [(p, g) for p, g, t in ps if t == tag]
        res[f"acc[{tag}]"] = round(sum((p >= .5) == g for p, g in sub) / len(sub), 3)
    return res


def task_link(c, pages, rng, n):
    names = list(pages)
    pos_cases = []
    for name, pg in pages.items():
        for para in paragraphs(pg["body"]) or []:
            for m in LINK.finditer(para):
                tgt = m.group(2)[2:]
                if tgt in pages and tgt != name:
                    pos_cases.append((name, LINK.sub(r"\1", para), tgt))
    rng.shuffle(pos_cases)
    pos_cases = pos_cases[:n]
    cases = []
    for name, snippet, tgt in pos_cases:
        t = pages[tgt]
        cases.append((f"{snippet}\n\n候補ページ: {t['title']} — {t['summary']}", True, "pos"))
        linked = {m.group(2)[2:] for m in LINK.finditer(pages[name]["body"])} | {name, tgt}
        pool_hard = [x for x in names if x not in linked and pages[x]["tags"] & t["tags"]]
        pool_easy = [x for x in names if x not in linked and not (pages[x]["tags"] & t["tags"])]
        for kind, pool in (("neg_hard", pool_hard), ("neg_easy", pool_easy)):
            if pool:
                o = pages[rng.choice(pool)]
                cases.append((f"{snippet}\n\n候補ページ: {o['title']} — {o['summary']}", False, kind))
    q = "この文章は、候補ページで説明されている概念に言及しており、そのページへリンクするのが適切ですか？"
    return len(pos_cases), run_binary(c, cases, q)


def task_sense(c, pages, rng, n):
    same, diff = [], []
    by_tag = {}
    for name, pg in pages.items():
        for t in pg["tags"]:
            by_tag.setdefault(t, []).append(name)
    for name, pg in pages.items():
        ps = paragraphs(pg["body"])
        for a, b in zip(ps, ps[1:]):
            same.append((a, b))
        sibs = [x for t in pg["tags"] for x in by_tag[t] if x != name]
        if ps and sibs:
            other = paragraphs(pages[rng.choice(sibs)]["body"])
            if other:
                diff.append((rng.choice(ps), rng.choice(other)))
    rng.shuffle(same); rng.shuffle(diff)
    cases = [(f"段落1:\n{a}\n\n段落2:\n{b}", True, "same_page") for a, b in same[:n]] + \
            [(f"段落1:\n{a}\n\n段落2:\n{b}", False, "sibling_pages") for a, b in diff[:n]]
    q = "段落1と段落2は、同じ一つの概念について述べていますか？"
    return len(cases), run_binary(c, cases, q)


def task_type(c, pages, rng, n, types):
    items = [(p["title"], p["summary"], p["type"]) for p in pages.values() if p["type"] in types and p["summary"]]
    rng.shuffle(items)
    labels = {"concept": "一つの概念(1センス)の定義を説明するページ", "synthesis": "複数の対象の対比・分析・接続をまとめるページ",
              "preference": "ユーザーの好みや方針を記録するページ", "thinking": "考察や検討中の考えを記録するページ",
              "feedback": "ユーザーからの指摘・フィードバックの記録", "idea": "思いつき・アイデアの記録",
              "judgment": "判断とその理由の記録", "learning": "実際に学んだことの記録", "capture": "外部の文章を取り込んだ記録"}
    crit = {t: labels[t] for t in types}
    pairs, errs, conf = [], 0, {}
    for title, summ, gold in items[:n]:
        resp, _, _ = c.ask(MLX, f"タイトル: {title}\n要約: {summ}", {"q": {"type": "choice", "instructions": "このページの種別はどれですか？", "criteria": crit}})
        if resp is None:
            errs += 1
            continue
        a = resp["answers"]["q"]
        pairs.append((a["confidence"], a["choice"] == gold))
        conf[(gold, a["choice"])] = conf.get((gold, a["choice"]), 0) + 1
    res = summarize(pairs, errs)
    res["majority_baseline"] = round(max(sum(1 for i in items[:n] if i[2] == t) for t in types) / max(1, len(items[:n])), 3)
    res["confusion"] = {f"{g}->{p}": v for (g, p), v in sorted(conf.items())}
    return len(items[:n]), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--base", default="http://127.0.0.1:8017")
    a = ap.parse_args()
    root, c, rng = Path(a.root), Client(a.base), random.Random(0)
    know = load(root, "knowledge")
    memos = load(root, "memos")
    out = {"pages": {"knowledge": len(know), "memos": len(memos)}}
    for name, fn in (("link", lambda: task_link(c, know, rng, a.n)),
                     ("sense", lambda: task_sense(c, know, rng, a.n)),
                     ("type_knowledge", lambda: task_type(c, know, rng, a.n, ["concept", "synthesis", "preference", "thinking"])),
                     ("type_memos", lambda: task_type(c, memos, rng, a.n, ["feedback", "idea", "judgment", "learning", "capture"]))):
        n, res = fn()
        out[name] = {"cases": n, **res}
        print(name, json.dumps(out[name], ensure_ascii=False), flush=True)
    Path(__file__).with_name(result_name("results_wiki.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
