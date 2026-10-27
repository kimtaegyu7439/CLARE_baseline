#!/usr/bin/env python
"""LOTUS lifelong setting 을 실제 LeRobot v2.1 데이터셋으로 물리 생성한다.

입력  /home/sa090180/Datasets/lerobot/LOTUS_setting/lotus_setting.json  (매니페스트)
출력  같은 디렉터리 아래 실험/스플릿마다 독립 데이터셋

    LOTUS_setting/LIBERO-50/base/          25 task x 50 demo
    LOTUS_setting/LIBERO-50/stage1..5/      5 task x 10 demo
    LOTUS_setting/LIBERO-OBJECT/base/       6 task x 50 demo
    LOTUS_setting/LIBERO-OBJECT/stage1..4/  1 task x 10 demo
    LOTUS_setting/LIBERO-GOAL/  (동일)

왜 복사인가 (심볼릭 링크가 안 되는 이유)
    parquet 안에 episode_index / index / task_index 가 **값으로** 저장돼 있다.
    새 데이터셋은 에피소드를 0 부터 다시 매기므로 그 세 열을 고쳐 써야 하고,
    따라서 파일을 새로 쓸 수밖에 없다. frame_index 는 에피소드 내 번호라 그대로 둔다.

    index 는 데이터셋 전체를 관통하는 프레임 일련번호라 누적으로 다시 센다.

    episodes_stats.jsonl 도 위 세 열의 통계만 다시 계산하고 나머지(이미지/상태/
    액션/타임스탬프/frame_index)는 원본 값을 그대로 옮긴다 — 내용이 안 변하니까.

    python lotus_build_setting.py                # 전부
    python lotus_build_setting.py --only LIBERO-50
    python lotus_build_setting.py --dry_run      # 계획만 출력
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# 경로는 환경변수를 먼저 보고(다른 서버), 없으면 서버1 의 기존 값을 쓴다.
#   LOTUS_DATA_ROOT  = LOTUS_setting 디렉터리
#   HF_LEROBOT_HOME  = 소스 데이터셋들이 있는 lerobot 루트 (bash/clare/env.sh 가 준다)
_SERVER1_LEROBOT = "/home/sa090180/Datasets/lerobot"
LEROBOT_HOME = Path(os.environ.get("HF_LEROBOT_HOME", _SERVER1_LEROBOT))
ROOT = Path(os.environ.get("LOTUS_DATA_ROOT") or LEROBOT_HOME / "LOTUS_setting")
MANIFEST = ROOT / "lotus_setting.json"


def resolve_source(path: str) -> str:
    """매니페스트의 소스 경로는 서버1 절대경로다. 그 경로가 없으면 접두사를
    이 서버의 HF_LEROBOT_HOME 으로 바꿔 본다 (예: /home/jovyan/Datasets/lerobot/...)."""
    if Path(path).exists() or not path.startswith(_SERVER1_LEROBOT + "/"):
        return path
    alt = LEROBOT_HOME / path[len(_SERVER1_LEROBOT) + 1:]
    if alt.exists():
        return str(alt)
    raise SystemExit(f"소스 데이터셋이 없다: {path}  (대체 경로도 없음: {alt})")
CHUNK = 1000
REINDEXED = ("episode_index", "index", "task_index")


def load_source(path: str, cache: dict) -> dict:
    """소스 데이터셋의 meta 를 한 번만 읽어 캐시한다."""
    if path in cache:
        return cache[path]
    p = Path(path)
    info = json.loads((p / "meta" / "info.json").read_text())
    eps = {json.loads(l)["episode_index"]: json.loads(l)
           for l in (p / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()}
    sp = p / "meta" / "episodes_stats.jsonl"
    st = {}
    if sp.exists():
        st = {json.loads(l)["episode_index"]: json.loads(l)["stats"]
              for l in sp.read_text().splitlines() if l.strip()}
    cache[path] = {"root": p, "info": info, "eps": eps, "stats": st}
    return cache[path]


def col_stats(v: list[int]) -> dict:
    n = len(v)
    mean = sum(v) / n
    var = sum((x - mean) ** 2 for x in v) / n
    return {"min": [min(v)], "max": [max(v)], "mean": [mean],
            "std": [var ** 0.5], "count": [n]}


def build_split(name: str, out: Path, tasks: list[dict], cache: dict,
                dry: bool) -> dict:
    """tasks = 매니페스트의 태스크 엔트리 목록. 순서대로 에피소드를 이어 붙인다."""
    # ★ 태스크 유일화 기준은 **지시문 문자열이 아니라 태스크 엔트리**다.
    #   LIBERO 에는 장면이 다른데 지시문이 같은 태스크가 있다(예: KITCHEN_SCENE10 과
    #   SCENE5 의 "close the top drawer of the cabinet"). 문자열로 묶으면 서로 다른
    #   태스크가 한 task_index 로 합쳐져 태스크별 평가가 불가능해진다.
    #   문자열은 원본 그대로 두므로 정책이 받는 **언어 조건은 바뀌지 않는다** —
    #   달라지는 것은 정수 id 뿐이다. LeRobot 읽기 경로는 index->문자열 방향만 쓰므로
    #   (lerobot_dataset.py:861) tasks.jsonl 에 같은 문자열이 두 줄 있어도 문제없다.
    plan = []                                   # (src_path, src_ep, instruction, task_ord)
    for ti, t in enumerate(tasks):
        a, b = t["episode_range"]
        plan += [(resolve_source(t["dataset"]), e, t["instruction"], ti) for e in range(a, b)]

    instrs = [t["instruction"] for t in tasks]              # 태스크 수 == len(tasks)
    n_ep, n_fr = len(plan), 0
    if dry:
        load_source(plan[0][0], cache)
        return {"split": name, "episodes": n_ep, "tasks": len(instrs)}

    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    ep_lines, st_lines, feat = [], [], None

    for j, (src, se, instr, ti) in enumerate(plan):
        S = load_source(src, cache)
        feat = feat or S["info"]["features"]
        f = S["root"] / S["info"]["data_path"].format(
            episode_chunk=se // S["info"]["chunks_size"], episode_index=se)
        tbl = pq.read_table(f)
        n = tbl.num_rows
        new = {"episode_index": [j] * n,
               "index": list(range(n_fr, n_fr + n)),
               "task_index": [ti] * n}
        for c, vals in new.items():
            tbl = tbl.set_column(tbl.schema.get_field_index(c), c,
                                 pa.array(vals, type=tbl.schema.field(c).type))
        d = out / f"data/chunk-{j // CHUNK:03d}"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, d / f"episode_{j:06d}.parquet")

        ep_lines.append({"episode_index": j, "tasks": [instr], "length": n})
        s = dict(S["stats"].get(se, {}))
        if s:
            for c in REINDEXED:
                s[c] = col_stats(new[c])
            st_lines.append({"episode_index": j, "stats": s})
        n_fr += n

    src0 = load_source(plan[0][0], cache)["info"]
    info = {"codebase_version": src0["codebase_version"], "robot_type": src0["robot_type"],
            "total_episodes": n_ep, "total_frames": n_fr, "total_tasks": len(instrs),
            "total_videos": 0, "total_chunks": (n_ep - 1) // CHUNK + 1,
            "chunks_size": CHUNK, "fps": src0["fps"],
            "splits": {"train": f"0:{n_ep}"},
            "data_path": src0["data_path"], "video_path": src0["video_path"],
            "features": feat}
    m = out / "meta"
    (m / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    (m / "tasks.jsonl").write_text("".join(
        json.dumps({"task_index": k, "task": s}, ensure_ascii=False) + "\n"
        for k, s in enumerate(instrs)))
    (m / "episodes.jsonl").write_text("".join(
        json.dumps(e, ensure_ascii=False) + "\n" for e in ep_lines))
    if st_lines:
        (m / "episodes_stats.jsonl").write_text("".join(
            json.dumps(e) + "\n" for e in st_lines))
    return {"split": name, "episodes": n_ep, "frames": n_fr, "tasks": len(instrs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="실험 하나만 (예: LIBERO-50)")
    ap.add_argument("--split", default=None,
                    help="스플릿 하나만 (예: stage1). 나머지는 손대지 않는다 — "
                         "중단된 스플릿만 복구할 때 쓴다.")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    man = json.loads(MANIFEST.read_text())
    cache: dict = {}
    for exp, ex in man["experiments"].items():
        if a.only and exp != a.only:
            continue
        print(f"\n══ {exp}")
        splits = [("base", ex["base"])] + [
            (f"stage{s['stage']}", s["tasks"]) for s in ex["lifelong_stages"]]
        for nm, tasks in splits:
            if a.split and nm != a.split:
                continue
            r = build_split(f"{exp}/{nm}", ROOT / exp / nm, tasks, cache, a.dry_run)
            print(f"   {nm:<8} 에피소드 {r['episodes']:>5}  태스크 {r['tasks']:>3}"
                  + (f"  프레임 {r['frames']:>7}" if "frames" in r else "  (dry)"))
    print("\n->", ROOT)


if __name__ == "__main__":
    main()
