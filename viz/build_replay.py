"""Build the replay page (STORM / OVO / SoccerNet / MemInStream) from the STORM replay page plus the OVO-Bench samples.

Keeps the original STORM page untouched. Adds one clip per OVO video, converts OVO's forward-active
tasks (REC / SSR / CRR) into one question per trigger time, and shows the Baseline A (P1) pick of
Qwen3-VL-8B on every question that has a result.

Usage: python viz/build_replay.py
"""
import base64, glob, json, os, re, subprocess, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probes", "storm"))
import probe
from probe import LETTERS, permutation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data", "storm", "STORM-Bench 流式回放.html")  # original STORM page, used as the template
OUT = os.path.join(HERE, "流式回放 STORM · OVO · SoccerNet.html")
HAS_SN = os.path.isdir(os.path.join(ROOT, "probes", "soccernet"))  # SoccerNet probe removed 2026-11-24; kept optional
OVO = os.path.join(ROOT, "data", "ovo")

LABELS = {
    "OpenEQA/036-hm3d-rJhMRvNn4DS.mp4": "OpenEQA 室内漫游",
    "Ego4D/clips/1945d70d-bb9d-4523-aa31-7cfedb76b6c6.mp4": "Ego4D 第一人称",
    "AutoEvalMetaData/107.mp4": "网络视频 107",
    "AutoEvalMetaData/243.mp4": "网络视频 243",
    "AutoEvalMetaData/242.mp4": "网络视频 242",
    "Ego4D/video/15929976-8630-46c5-ae6b-84a4dacd2634.mp4": "Ego4D 长视频",
    "COIN/Hf2AisK1wHY.mp4": "COIN 换轮胎",
    "perception_test/video_1696.mp4": "Perception Test",
}
GROUP = {"EPM": "后向", "ASI": "后向", "HLD": "后向", "STU": "实时", "OJR": "实时", "ATR": "实时",
         "ACR": "实时", "OCR": "实时", "FPD": "实时", "REC": "前向", "SSR": "前向", "CRR": "前向"}
ABSTAIN = re.compile(r"unable to answer|cannot be determined|uncertain|unresolved|no definite|"
                     r"not uniquely|no concrete alternative", re.I)


def b64(path, mime):
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def ffprobe_dur(v):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", v],
                                capture_output=True, text=True).stdout)


def encode(v, tmp):
    out = os.path.join(tmp, "v.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v, "-an", "-vf", "scale=480:-2,fps=15",
                    "-c:v", "libx264", "-crf", "32", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", out], check=True)
    d = os.path.join(tmp, "th"); os.makedirs(d, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v, "-vf", "fps=1,scale=-2:60", "-q:v", "6",
                    os.path.join(d, "%05d.jpg")], check=True)
    thumbs = [b64(p, "image/jpeg") for p in sorted(glob.glob(os.path.join(d, "*.jpg")))]
    return b64(out, "video/mp4"), thumbs


def load_preds(path, key):
    """id -> option index picked by P1, mapped back through the fixed permutation."""
    preds = {}
    if not os.path.exists(path):
        return preds
    for r in map(json.loads, open(path)):
        if r["probe"] != "P1" or not r["pred"] or r["pred"] not in LETTERS:
            continue
        qid = str(r[key])
        preds[qid] = r
    return preds


def ovo_questions(q, preds):
    t = q["task"]
    if "options" in q:  # multiple choice
        p = preds.get(str(q["id"]))
        pred_index = None
        if p:
            perm = permutation(str(q["id"]), len(q["options"]))
            pred_index = perm[LETTERS.index(p["pred"])]
        unc = bool(ABSTAIN.search(q["options"][q["gt"]]))
        return [{"id": f"ovo_{q['id']}", "query_time": q["realtime"], "question_type": t,
                 "question": q["question"], "options": q["options"], "answer_index": q["gt"],
                 "change_intensity": None, "status": "uncertain" if unc else "known", "sources": [],
                 "evidence_spans": [], "pred_index": pred_index, "group": GROUP[t]}]
    out = []
    for k, ti in enumerate(q["test_info"]):
        rt = ti["realtime"]
        if t == "REC":
            spans = [[a, b] for a, b in zip(q["start_times"], q["end_times"]) if a <= rt]
            n = ti["count"]
            opts = [str(x) for x in range(max(0, n - 1), n + 3)]
            out.append({"question": f"How many times has the action \"{q['activity']}\" happened so far?", "options": opts,
                        "answer_index": opts.index(str(n)), "evidence_spans": spans, "query_time": rt})
        elif t == "SSR":
            i = q["all_steps"].index(ti["step"])
            out.append({"question": f"Is the person currently performing the step \"{ti['step']}\"?", "options": ["No", "Yes"],
                        "answer_index": ti["type"], "evidence_spans": [[q["start_time"][i], q["end_time"][i]]],
                        "query_time": rt})
        elif t == "CRR":
            out.append({"question": f"{q['question']} Is there enough information to answer this question now? "
                                    f"(Reference answer: {q['answer']})", "options": ["No", "Yes"],
                        "answer_index": ti["type"], "evidence_spans": [[q["clue_time"], q["clue_time"]]],
                        "query_time": rt})
        out[-1].update({"id": f"ovo_{q['id']}_t{k}", "question_type": t, "change_intensity": None,
                        "status": "known", "sources": [], "pred_index": None, "group": GROUP[t]})
    return out


SN_ZH = {"Penalty": "点球", "Kick-off": "开球", "Goal": "进球", "Substitution": "换人", "Offside": "越位",
         "Shots on target": "射正", "Shots off target": "射偏", "Clearance": "解围", "Ball out of play": "球出界",
         "Throw-in": "界外球", "Foul": "犯规", "Indirect free-kick": "间接任意球", "Direct free-kick": "直接任意球",
         "Corner": "角球", "Yellow card": "黄牌", "Red card": "红牌", "Yellow->red card": "两黄变红"}


def soccernet_clips():
    """One clip per SoccerNet segment: labelled events in the feed (hit or missed by the polling baseline)
    plus the baseline's false alarms."""
    sys.path.insert(0, os.path.join(ROOT, "probes", "soccernet"))
    from sn_spot import DATA as SN, SEGMENTS, WINDOW, seg_id
    import sn_analyze
    res = os.path.join(ROOT, "probes", "soccernet", "results", "sn_qwen3vl8b.jsonl")
    polls = [json.loads(l) for l in open(res)] if os.path.exists(res) else []
    sn_analyze.polls = polls
    gts = sn_analyze.gt_events() if polls else []
    hits = {}
    if polls:  # rerun the matching to know which event each poll matched
        preds = sorted(({"segment": p["segment"], "label": c, "t": p["t"]} for p in polls for c in p["pred"]),
                       key=lambda x: x["t"])
        used = set()
        for g in sorted(gts, key=lambda x: x["t"]):
            for i, p in enumerate(preds):
                if i not in used and p["segment"] == g["segment"] and p["label"] == g["label"] \
                        and g["t"] - sn_analyze.SLACK <= p["t"] <= g["t"] + sn_analyze.TOL:
                    used.add(i); hits[(g["segment"], g["label"], g["t"])] = p["t"]; break
        false_alarms = [p for i, p in enumerate(preds) if i not in used]
    clips = []
    for game, half, a, b in SEGMENTS:
        sid = seg_id(game, half, a, b)
        lo = max(0, a - WINDOW)
        video = os.path.join(SN, "videos", game, f"{half}_224p.mkv")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "v.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(lo), "-i", video, "-t", str(b - lo), "-an",
                            "-vf", "fps=15", "-c:v", "libx264", "-crf", "32", "-preset", "veryfast",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
            vdata = b64(out, "video/mp4")
        fdir = os.path.join(SN, "frames", sid)
        thumbs = [b64(p, "image/jpeg") for p in sorted(glob.glob(os.path.join(fdir, "*.jpg")))[: b - lo + 1]]
        qs = []
        polled = {p["t"] for p in polls if p["segment"] == sid}
        for g in (x for x in gts if x["segment"] == sid):
            if g["t"] > max(polled, default=0):
                continue
            h = hits.get((sid, g["label"], g["t"]))
            note = (f"Baseline 在 +{h - g['t']:.0f}s 报出" if h is not None else "Baseline 未报出")
            qs.append({"id": f"{sid}_{g['label']}_{g['t']}", "query_time": round(g["t"] - lo, 1), "question_type": "SN",
                       "question": f"{g['label']}（{SN_ZH.get(g['label'], '')}）", "options": [], "answer_index": -1,
                       "change_intensity": None, "status": "known" if g["visible"] else "uncertain",
                       "chip": "画面可见" if g["visible"] else "画面未显示", "sources": [],
                       "evidence_spans": [[round(g["t"] - lo, 1), round(g["t"] - lo + sn_analyze.TOL, 1)]],
                       "pred_index": None, "note": note, "note_ok": h is not None})
        for p in (x for x in (false_alarms if polls else []) if x["segment"] == sid):
            qs.append({"id": f"{sid}_fa_{p['label']}_{p['t']}", "query_time": p["t"] - lo, "question_type": "SN_FP",
                       "question": f"{p['label']}（{SN_ZH.get(p['label'], '')}）", "options": [], "answer_index": -1,
                       "change_intensity": None, "status": "uncertain", "chip": "误报", "sources": [],
                       "evidence_spans": [[p["t"] - lo - 5, p["t"] - lo]], "pred_index": None,
                       "note": "附近 10 秒内没有这一类事件的标注", "note_ok": False})
        name = re.sub(r"^\S+ - \S+ ", "", game.split("/")[-1])
        clips.append({"key": "sn_" + re.sub(r"\W+", "_", sid), "label": f"{name} {a // 60}′–{b // 60}′",
                      "track": "SoccerNet · 事件检测", "source": f"SoccerNet / {game.split('/')[-1]}，第 {half} 半场",
                      "duration": float(b - lo), "video": vdata, "webm": "", "thumbs": thumbs, "questions": qs,
                      "bench": "sn"})
        print(clips[-1]["label"], len(qs), "items")
    return clips


MIS = os.path.join(ROOT, "data", "meminstream")
MIS_RES = os.path.join(ROOT, "probes", "ovo", "results", "meminstream_qwen3vl8b.jsonl")
MIS_L1 = {"Episodic Retrieval": "情景回忆", "Multi-Evidence Integration": "多证据整合",
          "State Tracking & Updating": "状态追踪", "Contextual Memory": "上下文记忆"}
MIS_CLIPS = 12  # keep the page size reasonable: the clips with the most sampled questions, state tracking first


def meminstream_clips():
    """MemInStream sample (probes/meminstream/fetch.py): annotated evidence spans + the Baseline A (P1) pick."""
    import hashlib
    rows = [json.loads(l) for l in open(os.path.join(MIS, "sample.jsonl"))]
    rows = [r for r in rows if isinstance(r["correct_index"], int)]  # skip the few multi-answer questions
    preds = {}
    if os.path.exists(MIS_RES):
        for r in map(json.loads, open(MIS_RES)):
            if r["probe"] == "P1" and r["pred"]:
                preds[r["id"]] = r["pred"]
    by = {}
    for r in rows:
        by.setdefault(r["local"], []).append(r)
    order = sorted(by, key=lambda v: (not any(r["level1"] == "State Tracking & Updating" for r in by[v]), -len(by[v]), v))
    picked, seen_l1 = [], set()
    for v in order:  # cover every level-1 category, then fill up
        if len(picked) < MIS_CLIPS and ({r["level1"] for r in by[v]} - seen_l1 or len(seen_l1) == len(MIS_L1)):
            picked.append(v); seen_l1 |= {r["level1"] for r in by[v]}
    clips = []
    for v in picked:
        qs = []
        for r in sorted(by[v], key=lambda r: r["query_time"]):
            qid = "mis" + hashlib.md5((r["video_id"] + r["question"]).encode()).hexdigest()[:10]
            opts = [re.sub(r"^[A-E]\.\s*", "", o) for o in r["options"]]
            pred_index = None
            if qid in preds:
                perm = permutation(qid, len(opts))
                pred_index = perm["ABCDE".index(preds[qid])]
            qs.append({"id": qid, "query_time": r["query_time"], "question_type": r["level2"],
                       "question": r["question"], "options": opts, "answer_index": r["correct_index"],
                       "change_intensity": None, "status": "known", "sources": [],
                       "evidence_spans": r["evidence"], "pred_index": pred_index, "group": MIS_L1[r["level1"]]})
        path = os.path.join(MIS, "videos", v)
        with tempfile.TemporaryDirectory() as tmp:
            video, thumbs = encode(path, tmp)
        src = by[v][0].get("source_video_dataset") or "未标注来源"
        clips.append({"key": "mis_" + hashlib.md5(v.encode()).hexdigest()[:8],
                      "label": f"{by[v][0].get('Area') or src} #{len(clips) + 1}",
                      "track": "MemInStream · " + "/".join(sorted({q["group"] for q in qs})),
                      "source": f"MemInStream / {src}", "duration": round(ffprobe_dur(path), 2),
                      "video": video, "webm": "", "thumbs": thumbs, "bench": "mis", "questions": qs})
        print("MemInStream", clips[-1]["label"], len(qs), "questions")
    return clips


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2


def summaries():
    """Headline Baseline numbers per benchmark, computed from the result files."""
    out = {}
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "probes", "storm", "results", "qwen3vl8b.jsonl"))]
    real = {"storm_cook", "storm_bike", "storm_healthy", "storm_music", "storm_sports"}
    pick = lambda p, s: [r for r in rows if r["probe"] == p and r["status"] == s and r["subset"] in real]
    acc = lambda rs: f"{sum(r['correct'] for r in rs) / len(rs):.2f}"
    k1, unc = pick("P1", "known"), pick("P1", "uncertain")
    out["storm"] = {
        "title": "STORM-Bench · Baseline A（Qwen3-VL-8B 4-bit，1 帧/秒，最多 32 帧）",
        "stats": [[acc(k1), f"看全部历史帧（P1），可答题 {len(k1)} 道"],
                  [acc(pick("P0", "known")), "不看视频、只给选项（P0）"],
                  [acc(pick("P2", "known")), "只看标注的证据帧（P2）"],
                  [f"{sum(r['correct'] for r in unc)}/{len(unc)}", "不确定题里选了“无法确定”"],
                  [f"{median([r['sec'] for r in k1 + unc]):.0f} 秒", "每题延迟中位数（M4）"]],
        "note": "看全部历史和只看证据帧一样好：这些短视频（8–43 秒）上记忆没有提升空间，错题主要是看错了画面。"
                "模型几乎不会主动说“无法确定”。数字来自 5 个真实视频片段（厨房、修车、健身、音乐、体育），回放里只收录了厨房和修车；模拟片段（THOR、VHome）没有跑。"}
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "probes", "ovo", "results", "ovo_qwen3vl8b.jsonl"))]
    p1 = [r for r in rows if r["probe"] == "P1"]
    p0 = [r for r in rows if r["probe"] == "P0"]
    grp = lambda rs, g: [r for r in rs if GROUP[r["task"]] == g]
    out["ovo"] = {
        "title": "OVO-Bench 小样本 · Baseline A（同一设置）",
        "stats": [[acc(p1), f"看全部历史帧（P1），选择题 {len(p1)} 道"],
                  [acc(p0), "不看视频、只给选项（P0）"],
                  [f"{acc(grp(p1, '后向'))} / {acc(grp(p0, '后向'))}", f"后向题 {len(grp(p1, '后向'))} 道：P1 / P0"],
                  [f"{acc(grp(p1, '实时'))} / {acc(grp(p0, '实时'))}", f"实时题 {len(grp(p1, '实时'))} 道：P1 / P0"],
                  [f"{median([r['sec'] for r in p1]):.0f} 秒", "每题延迟中位数（M4）"]],
        "note": "不看视频也能答对六成，后向题盲猜就有 0.81：很多题靠常识就能猜，HLD 题的“无法回答”选项一出现就是答案，"
                "所以这批题测不出记忆。前向题（REC / SSR / CRR）只在回放里展示，没有打分。"}
    if os.path.exists(MIS_RES):
        rows = [json.loads(l) for l in open(MIS_RES)]
        by = lambda p: [r for r in rows if r["probe"] == p]
        st = lambda p: [r for r in by(p) if r["level1"] == "State Tracking & Updating"]
        out["mis"] = {
            "title": f"MemInStream 小样本 · 探针 P0–P3（Qwen3-VL-8B 4-bit，{len(by('P1'))} 题）",
            "stats": [[acc(by("P1")), "看全部历史（P1，32 帧均匀铺开）"],
                      [acc(by("P2")), "只看标注的证据帧（P2，最多 8 帧）"],
                      [acc(by("P0")), "不看视频、只给选项（P0）"],
                      [acc(by("P3")), "证据帧先写笔记再答（P3）"],
                      [f"{acc(st('P1'))} / {acc(st('P2'))}", f"状态追踪题 {len(st('P1'))} 道：P1 / P2"]],
            "note": "每道题都标了证据片段，P2 直接用标注；P2 与 P1 的差距就是“记忆有没有用武之地”。"
                    "样本按 17 个小类各取 3 题，偏向较小的 YouTube 视频（提问时刻中位数 170 秒）；"
                    "回放里只收录了其中 12 个片段。"}
    if not HAS_SN:
        return out
    sys.path.insert(0, os.path.join(ROOT, "probes", "soccernet"))
    import sn_analyze
    polls = [json.loads(l) for l in open(os.path.join(ROOT, "probes", "soccernet", "results", "sn_qwen3vl8b.jsonl"))]
    gts = [g for g in sn_analyze.gt_events() if g["t"] <= max(p["t"] for p in polls if p["segment"] == g["segment"])]
    preds = sorted(({"segment": p["segment"], "label": c, "t": p["t"]} for p in polls for c in p["pred"]), key=lambda x: x["t"])
    used = set()
    for g in sorted(gts, key=lambda x: x["t"]):
        g["hit"] = None
        for i, p in enumerate(preds):
            if i not in used and p["segment"] == g["segment"] and p["label"] == g["label"] \
                    and g["t"] - sn_analyze.SLACK <= p["t"] <= g["t"] + sn_analyze.TOL:
                used.add(i); g["hit"] = p["t"] - g["t"]; break
    vis = [g for g in gts if g["visible"]]
    hit = lambda gs: sum(g["hit"] is not None for g in gs)
    out["sn"] = {
        "title": "SoccerNet 事件检测 · 固定间隔轮询（Baseline B：每 5 秒看最近 10 秒）",
        "stats": [[f"{hit(vis) / len(vis):.2f}", f"画面可见事件的召回（{hit(vis)}/{len(vis)}）"],
                  [f"{hit(gts) / len(gts):.2f}", f"全部标注事件的召回（{hit(gts)}/{len(gts)}）"],
                  [f"{len(used) / len(preds):.2f}", f"精度：{len(preds)} 次报出里对了 {len(used)} 次"],
                  [f"{median([g['hit'] for g in gts if g['hit'] is not None]):.1f} 秒", "检出事件的报出延迟中位数"],
                  [f"{median([p['sec'] for p in polls]):.1f} 秒", "每次调用耗时中位数（间隔只有 5 秒）"]],
        "note": f"能报出一半左右看得见的事件，但误报太多：{len(polls)} 次轮询只有 {sum(not p['pred'] for p in polls)} 次回答“没有事件”，"
                "回放和庆祝画面常被报成进球。每次调用比轮询间隔还慢，M4 上跟不上实时。"
                f"打分：类名一致且在事件前 {sn_analyze.SLACK} 秒到后 {sn_analyze.TOL} 秒内报出算检出。"}
    return out


BENCH_UI = """
<div class="bench" id="bench" role="group" aria-label="选择 benchmark">
  <button data-b="mis">MemInStream</button><button data-b="storm">STORM-Bench</button><button data-b="ovo">OVO-Bench</button><button data-b="sn">SoccerNet</button>
</div>
<section class="res" id="res" aria-label="Baseline 结果"></section>
"""
BENCH_CSS = (
    ".bench{display:inline-flex;margin:24px 0 0}"
    ".bench button{font:inherit;font-size:.88rem;border:1px solid var(--line);background:transparent;color:var(--mid);padding:6px 16px;cursor:pointer;margin-left:-1px}"
    ".bench button[aria-pressed=\"true\"]{border-color:var(--ink);background:var(--ink);color:var(--bg);position:relative}"
    ".res{margin:18px 0 0;border-top:2px solid var(--ink);padding-top:10px}"
    ".res h2{font-size:.85rem;font-weight:600;margin:0 0 10px}"
    ".stats{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px 24px}"
    "@media (max-width:760px){.stats{grid-template-columns:repeat(2,minmax(0,1fr))}}"
    ".stats b{display:block;font-size:1.35rem;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.2}"
    ".stats span{font-size:.76rem;color:var(--mid)}"
    ".res p{font-size:.82rem;color:var(--mid);margin:10px 0 0;max-width:90ch}"
    "[hidden]{display:none!important}\n")
BENCH_JS = """
const BENCH = __BENCH__;
function selectBench(b){
  try{ localStorage.setItem("replay-bench", b); }catch(e){}
  document.querySelectorAll("#bench button").forEach(x=>x.setAttribute("aria-pressed", x.dataset.b===b));
  document.querySelectorAll("#clips button").forEach((x,i)=>x.hidden = DATA[i].bench!==b);
  const legend = document.querySelector(".ovo-cats"); if(legend) legend.hidden = b!=="ovo";
  const s = BENCH[b];
  $("#res").innerHTML = `<h2>${s.title}</h2><div class="stats">${s.stats.map(([n,l])=>`<div><b>${n}</b><span>${l}</span></div>`).join("")}</div><p>${s.note}</p>`;
  load(DATA.findIndex(c=>c.bench===b));
}
document.querySelectorAll("#bench button").forEach(x=>x.onclick=()=>selectBench(x.dataset.b));
let startBench = "mis"; try{ startBench = localStorage.getItem("replay-bench") || "mis"; }catch(e){}
selectBench(BENCH[startBench] ? startBench : "mis");
"""


def main():
    html = open(SRC, encoding="utf-8").read()
    m = re.search(r"const DATA = (\[.*?\]);\nconst TYPE", html, re.S)
    data = json.loads(m.group(1))
    for c in data:
        c["bench"] = "storm"

    # STORM clips: attach P1 picks from the pilot (real-video clips only were run)
    sp = load_preds(os.path.join(ROOT, "probes", "storm", "results", "qwen3vl8b.jsonl"), "id")
    for c in data:
        for q in c["questions"]:
            p = sp.get(q["id"])
            q["pred_index"] = permutation(q["id"])[LETTERS.index(p["pred"])] if p else None

    preds = load_preds(os.path.join(ROOT, "probes", "ovo", "results", "ovo_qwen3vl8b.jsonl"), "id")
    rows = [json.loads(l) for l in open(os.path.join(OVO, "samples.jsonl"))]
    for vid, label in LABELS.items():
        qs = [q for q in rows if q["video"] == vid]
        path = os.path.join(OVO, "videos", vid)
        with tempfile.TemporaryDirectory() as tmp:
            video, thumbs = encode(path, tmp)
        groups = sorted({GROUP[q["task"]] for q in qs}, key="后向实时前向".index)
        data.append({"key": "ovo_" + re.sub(r"\W+", "_", vid), "label": label, "track": "OVO · " + "/".join(groups),
                     "source": "OVO-Bench / " + vid.split("/")[0], "duration": round(ffprobe_dur(path), 2),
                     "video": video, "webm": "", "thumbs": thumbs, "bench": "ovo",
                     "questions": [x for q in qs for x in ovo_questions(q, preds)]})
        print(label, len(data[-1]["questions"]), "questions")

    if HAS_SN:
        data.extend(soccernet_clips())
    data[:0] = meminstream_clips()  # MemInStream first on the page

    html = html.replace(m.group(1), json.dumps(data, ensure_ascii=False))
    html = html.replace("<title>STORM-Bench 流式回放</title>", "<title>流式视频回放</title>")
    html = html.replace("<h1>STORM-Bench 流式回放</h1>", "<h1>流式视频回放：MemInStream · STORM · OVO"
                        + (" · SoccerNet" if HAS_SN else "") + "</h1>")
    # SoccerNet items: custom chip text and a note under the item
    html = html.replace("${q.status==='uncertain'?'不确定':'可答'}</span>", "${q.chip||(q.status==='uncertain'?'不确定':'可答')}</span>")
    html = html.replace(
        "点一道题，可以在时间轴上看到它的证据区间。</p>",
        "点一道题，可以在时间轴上看到它的证据区间。用下方按钮切换 benchmark，每个 benchmark 先列出 Baseline 的总体结果。OVO-Bench 片段按同样方式回放："
        "后向题考记忆，实时题考当下感知，前向题在多个时刻反复提问、要求模型等证据出现再答。"
        "每道题下方是 Baseline A（Qwen3-VL-8B，1 帧/秒，最多 32 帧）的选择。"
        + ("SoccerNet 片段是事件检测：每 5 秒让模型看最近 10 秒的画面，说出刚发生的事件；"
           "右侧列出标注的事件（是否在 10 秒内被报出）和模型的误报。" if HAS_SN else "")
        + "MemInStream 片段的每道题都带人工标注的证据区间，题目按提问时刻依次出现。</p>")
    legend = (
        '<section class="ovo-cats" aria-label="OVO-Bench 任务类别"><h2>OVO-Bench 的三类任务</h2><div class="cats">'
        '<div><b>后向 Backward Tracing</b><p>答案在过去，要回忆之前看到的东西。</p>'
        '<p class="tk">EPM 情节记忆 · ASI 动作序列 · HLD 幻觉检测（正确答案是“无法回答”）</p>'
        '<p class="ex">例：“我切茄子的时候在跟谁说话？”</p></div>'
        '<div><b>实时 Real-Time Perception</b><p>答案就在当下几秒的画面里。</p>'
        '<p class="tk">STU 时空 · OJR 物体 · ATR 属性 · ACR 动作 · OCR 文字 · FPD 未来预测</p>'
        '<p class="ex">例：“最大的巨人站在哪里？”</p></div>'
        '<div><b>前向 Forward Active Responding</b><p>提问时答案还没出现，模型要先等着，证据出现后再主动回答。</p>'
        '<p class="tk">REC 重复计数 · SSR 步骤识别 · CRR 线索响应</p>'
        '<p class="ex">例：在多个时刻反复问“现在能回答了吗？”</p></div>'
        '</div></section>')
    ui = BENCH_UI if HAS_SN else BENCH_UI.replace('<button data-b="sn">SoccerNet</button>', '')
    html = html.replace('<div class="clips" id="clips"', ui + legend + '\n<div class="clips" id="clips"', 1)
    html = html.replace("footer{", BENCH_CSS + "footer{", 1)
    assert "buildClips(); load(0);" in html
    html = html.replace("buildClips(); load(0);", "buildClips();\n" +
                        BENCH_JS.replace("__BENCH__", json.dumps(summaries(), ensure_ascii=False)))
    html = html.replace(
        "footer{",
        ".ovo-cats{margin:22px 0 0}.ovo-cats h2{font-size:.85rem;font-weight:600;margin:0 0 8px}"
        ".cats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0 24px;border-top:1px solid var(--line)}"
        "@media (max-width:760px){.cats{grid-template-columns:minmax(0,1fr)}}"
        ".cats>div{padding:10px 0;font-size:.8rem;color:var(--mid)}.cats b{color:var(--ink);font-weight:600}"
        ".cats p{margin:3px 0}.cats .tk{color:var(--ink)}\nfooter{", 1)
    # each OVO question shows its category next to its task
    html = html.replace("<span>${TYPE[q.question_type]||q.question_type}</span>",
                        "${q.group?`<span>${q.group}</span>`:''}<span>${TYPE[q.question_type]||q.question_type}</span>")
    # OVO has no change intensity / evidence for MCQ: render those parts only when present; show Baseline A pick
    html = html.replace("<span>${intensity(q.change_intensity)}</span>",
                        "${q.change_intensity!=null?`<span>${intensity(q.change_intensity)}</span>`:''}")
    html = html.replace(
        '<div class="evd">证据：${ev}</div></div></article>`;',
        '${ev?`<div class="evd">证据：${ev}</div>`:""}${q.pred_index!=null?`<div class="evd pred ${q.pred_index===q.answer_index?\'ok\':\'bad\'}">Baseline A（Qwen3-VL-8B）选 ${"ABCDE"[q.pred_index]}：${q.pred_index===q.answer_index?"对":"错"}</div>`:""}</div></article>`;')
    html = html.replace(".q .src{color:var(--unc)}",
                        ".q .src{color:var(--unc)}\n.q .pred.ok{color:var(--known)}\n.q .pred.bad{color:var(--unc)}\n.hideans .q .pred{display:none}")
    html = html.replace("${q.pred_index!=null?", "${q.note?`<div class=\"evd pred ${q.note_ok?'ok':'bad'}\">${q.note}</div>`:''}${q.pred_index!=null?", 1)
    TYPE_ADD = ('EPM:"情节记忆",ASI:"动作序列",HLD:"幻觉检测",STU:"时空理解",OJR:"物体识别",ATR:"属性识别",'
                'ACR:"动作识别",OCR:"文字识别",FPD:"未来预测",REC:"重复计数",SSR:"步骤识别",CRR:"线索响应",'
                'SN:"标注事件",SN_FP:"模型报出",'
                '"Temporal Fact Recall":"时间事实回忆","Action / Event Recall":"动作事件回忆","Attribute Recall":"属性回忆",'
                '"Spatial / Location Recall":"位置回忆","Entity / Object Recall":"物体回忆","Occurrence Verification":"是否出现过",'
                '"Counting":"计数","Temporal Ordering":"时间顺序","Cross-Time Comparison":"跨时间比较",'
                '"Sequential Action Integration":"连续动作整合","Multi-hop Memory Binding":"多跳记忆关联",'
                '"Latest-State Retrieval":"最新状态","Identity & State Persistence":"身份与状态保持",'
                '"State Transition Tracking":"状态转变","Correction / Revision Memory":"修正记忆",'
                '"Conversational Context Memory":"对话上下文","Entity–Identity Binding":"实体身份关联",')
    html = html.replace("const TYPE = {", "const TYPE = {" + TYPE_ADD, 1)
    html = html.replace(
        "标注沿用 HD-EPIC、Ego-Exo4D 等源数据集的许可条款。</footer>",
        "标注沿用 HD-EPIC、Ego-Exo4D 等源数据集的许可条款。OVO 片段与标注取自 "
        '<a href="https://github.com/JoeLeelyf/OVO-Bench">OVO-Bench</a>（视频来自 Hugging Face 数据集 JoeLeelyf/OVO-Bench），'
        "同样压缩到 480 像素宽；前向任务按每个触发时刻拆成一道题。"
        'MemInStream 片段与标注取自 <a href="https://github.com/HKUST-LongGroup/MemInStream">MemInStream</a>'
        "（Hugging Face 数据集 zizi32/MemInStream）。</footer>")
    open(OUT, "w", encoding="utf-8").write(html)
    print("wrote", OUT, round(os.path.getsize(OUT) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
