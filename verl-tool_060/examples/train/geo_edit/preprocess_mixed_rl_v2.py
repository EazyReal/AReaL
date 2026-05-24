"""Assemble the v2-0524 mixed RL parquets.

Reads three already-converted train parquets (full ReasonMap incl. SFT,
v2 visual_probe/map_trace/deep_eyes, OmniSpatial train) and concatenates them
into a single train parquet.  Builds a stratified 1/10 OmniSpatial test split
into a fresh val parquet and concatenates it with the existing ReasonMap 10%
test and the new_val (visual_probe + map_trace) parquet to produce the final
val parquet.

Compared to ``preprocess_mixed_rl.py`` (the v0420 build):
  * MapQA is dropped from both train and val.
  * ReasonMap uses the FULL combined_train.parquet (3266 rows = RL + SFT)
    instead of combined_train_rl_only.parquet (2152 rows).
  * OmniSpatial train (6885 rows from the existing converted parquet) is added
    to train.
  * OmniSpatial test (1533 rows) is stratified-sampled at 1/10 per category
    (Perspective_Taking 56 / Dynamic_Reasoning 42 / Spatial_Interaction 30 /
    Complex_Logic 25 = 153) for val.

Usage:
    python verl-tool_060/examples/train/geo_edit/preprocess_mixed_rl_v2.py
"""

from __future__ import annotations

import argparse
import io
import math
import random
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Features, Image as HFImage, Sequence, Value
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SEED = 42

# --- Source parquets (already in verl-tool schema) -------------------------
SRC_REASONMAP_TRAIN = "/storage/openpsi/data/reasonmap_rl/combined_train.parquet"
SRC_REASONMAP_VAL_10PCT = "/storage/openpsi/data/reasonmap_rl/combined_test_10pct.parquet"
SRC_NEW_TRAIN_V2 = "/storage/openpsi/data/lcy_image_edit/mixed_rl_v2/new_train.parquet"
SRC_NEW_VAL = "/storage/openpsi/data/lcy_image_edit/mixed_rl/new_val.parquet"
SRC_OMNISPATIAL_TRAIN_VT = (
    "/storage/openpsi/data/lcy_image_edit/mixed_rl_v2/omnispatial_rl_train.parquet"
)

# --- OmniSpatial test source (NOT yet in verl-tool schema) -----------------
SRC_OMNISPATIAL_TEST = "/storage/openpsi/data/lcy_image_edit/OmniSpatial/omnispatial.parquet"

# --- Output dir ------------------------------------------------------------
OUT_DIR = Path("/storage/openpsi/data/lcy_image_edit/mixed_rl_v2")
OUT_TRAIN = OUT_DIR / "train_v2_0524.parquet"
OUT_VAL = OUT_DIR / "val_v2_0524.parquet"
OUT_OMNISPATIAL_VAL = OUT_DIR / "omnispatial_rl_val_0524.parquet"

# --- verl-tool schema (must match preprocess_mixed_rl.py exactly) ----------
FEATURES = Features(
    {
        "data_source": Value("string"),
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "images": Sequence(HFImage()),
        "reward_model": {
            "style": Value("string"),
            "ground_truth": Value("string"),
            "extra": {
                "type": Value("string"),
                "station_1": Value("string"),
                "station_2": Value("string"),
                "metro_data": Value("string"),
            },
        },
        "extra_info": {
            "id": Value("string"),
            "city": Value("string"),
            "type": Value("string"),
        },
    }
)

# --- Prompt template (cloned 1:1 from preprocess_mixed_rl.py /
#                     existing omnispatial_rl_train.parquet) ---------------
USER_PROMPT_TEMPLATE = """\
Please answer the following {task_type} question:
Question: {question}
Please provide a complete step-by-step solution to
this problem. Your reasoning should:
1. Analyze the problem systematically
2. Check if the tool execution and answer are correct
3. If there are errors, explain what went wrong and
provide the correct reasoning
4. Provide the final answer
Use natural expressions like 'let me think' or 'hmm'
when helpful, but keep it concise. It's encouraged
to use self-reflection or verification especially in the
verifying tool output in the reasoning process.
Provide your detailed reasoning between <think>
and </think> tags, then give your final answer
between <answer> and </answer> tags.
Output format: {output_format}"""

OMNISPATIAL_OUTPUT_FORMAT = (
    "You MUST respond with only the option letter (A, B, C, or D) "
    "in <answer></answer> tags."
)


def get_system_prompt() -> str:
    """Reuse the exact 12986-char system prompt baked into the existing
    OmniSpatial v2 parquet.  Avoids re-importing ``geo_edit.tool_definitions``
    (which is not always installed) and guarantees an identical prompt across
    all data_source rows produced by this script."""
    src = pq.read_table(SRC_OMNISPATIAL_TRAIN_VT, columns=["prompt"]).to_pandas()
    sys_msg = src["prompt"].iloc[0][0]
    assert sys_msg["role"] == "system"
    assert len(sys_msg["content"]) > 1000, "system prompt sanity check"
    return sys_msg["content"]


SYSTEM_PROMPT: str | None = None


def _build_user_msg(task_type: str, question: str, output_format: str) -> str:
    body = USER_PROMPT_TEMPLATE.format(
        task_type=task_type, question=question, output_format=output_format
    )
    return "Observation 0:\n<image>\n" + body


def decode_image(img_field) -> Image.Image:
    """Decode an image stored either inline (bytes), as a filesystem path, or
    as a HuggingFace ``{bytes, path}`` dict (in OmniSpatial test the bytes are
    None and ``path`` points to the on-disk PNG)."""
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    if isinstance(img_field, dict):
        b = img_field.get("bytes")
        if b:
            return Image.open(io.BytesIO(b)).convert("RGB")
        p = img_field.get("path")
        if p:
            return Image.open(str(p)).convert("RGB")
        raise ValueError("dict image has neither bytes nor path")
    if isinstance(img_field, bytes):
        return Image.open(io.BytesIO(img_field)).convert("RGB")
    if isinstance(img_field, str):
        return Image.open(img_field).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(img_field)}")


def _verify_image_readable(img_field) -> None:
    decode_image(img_field).close()


def _passthrough_image(img_field):
    """Return an image reference suitable for an HFImage feature without
    re-encoding the pixels.  OmniSpatial test stores ``{bytes: None, path: ...}``
    and the existing ``omnispatial_rl_train.parquet`` keeps the same path-only
    form (9 MB for 6885 rows vs ~1 MB/row when bytes are inlined)."""
    if isinstance(img_field, dict):
        return {"bytes": img_field.get("bytes"), "path": img_field.get("path")}
    if isinstance(img_field, (bytes, bytearray)):
        return {"bytes": bytes(img_field), "path": None}
    if isinstance(img_field, str):
        return {"bytes": None, "path": img_field}
    raise ValueError(f"unsupported image field for passthrough: {type(img_field)}")


def make_omnispatial_record(row, system_prompt: str) -> dict:
    question_full = f"{row['question']}\n\n{row['options_text']}"
    user_msg = _build_user_msg(
        task_type="visual question answering",
        question=question_full,
        output_format=OMNISPATIAL_OUTPUT_FORMAT,
    )
    _verify_image_readable(row["image"])
    return {
        "data_source": "omnispatial",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "images": [_passthrough_image(row["image"])],
        "reward_model": {
            "style": "rule",
            "ground_truth": str(row["answer"]),
            "extra": {
                "type": str(row["category"]),
                "station_1": "",
                "station_2": "",
                "metro_data": "",
            },
        },
        "extra_info": {
            "id": str(row["index"]),
            "city": "",
            "type": str(row["category"]),
        },
    }


def stratified_sample_indices(categories: list[str], frac: float, seed: int) -> list[int]:
    """Return indices for a stratified random sample at ``frac`` per category.
    Uses ``round`` (not floor) per stratum so totals match expectation of
    ``len * frac`` to within rounding."""
    rng = random.Random(seed)
    by_cat: dict[str, list[int]] = {}
    for i, c in enumerate(categories):
        by_cat.setdefault(c, []).append(i)
    out: list[int] = []
    for c, idxs in by_cat.items():
        k = int(round(len(idxs) * frac))
        rng.shuffle(idxs)
        picked = sorted(idxs[:k])
        out.extend(picked)
    return sorted(out)


def build_omnispatial_val() -> Path:
    print("=== Building OmniSpatial val (1/10 stratified from test) ===")
    df = pq.read_table(SRC_OMNISPATIAL_TEST).to_pandas()
    print(f"  source rows: {len(df)}")
    print(f"  source category dist: {dict(Counter(df['category']).most_common())}")

    indices = stratified_sample_indices(df["category"].tolist(), frac=0.1, seed=SEED)
    sub = df.iloc[indices].reset_index(drop=True)
    print(f"  picked: {len(sub)} rows")
    print(f"  picked category dist: {dict(Counter(sub['category']).most_common())}")

    sys_prompt = get_system_prompt()
    records: list[dict] = []
    for _, r in sub.iterrows():
        try:
            records.append(make_omnispatial_record(r, sys_prompt))
        except Exception as e:
            print(f"  SKIP {r.get('index','?')}: {e}")

    OUT_OMNISPATIAL_VAL.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records, features=FEATURES).to_parquet(str(OUT_OMNISPATIAL_VAL))
    print(f"  wrote {len(records)} -> {OUT_OMNISPATIAL_VAL}")
    return OUT_OMNISPATIAL_VAL


def concat_parquets(srcs: list[str | Path], dst: Path, label: str) -> int:
    print(f"\n=== Building {label} -> {dst.name} ===")
    tables: list[pa.Table] = []
    total = 0
    for s in srcs:
        t = pq.read_table(str(s))
        print(f"  + {Path(s).name:40s} rows={t.num_rows}")
        tables.append(t)
        total += t.num_rows
    big = pa.concat_tables(tables, promote_options="permissive")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(big, str(dst))
    print(f"  -> wrote {big.num_rows} rows ({total} expected) to {dst}")
    return big.num_rows


def summarize(dst: Path, label: str) -> None:
    df = pq.read_table(dst, columns=["data_source", "extra_info"]).to_pandas()
    src_dist = Counter(df["data_source"]).most_common()
    type_dist: Counter = Counter()
    for ei in df["extra_info"]:
        t = ei.get("type", "") if isinstance(ei, dict) else ""
        type_dist[t] += 1
    print(f"\n--- Summary: {label} ({len(df)} rows) ---")
    print("  data_source:")
    for k, v in src_dist:
        print(f"    {k:30s}: {v}")
    print("  extra_info.type (top 10):")
    for k, v in type_dist.most_common(10):
        print(f"    {(repr(k))[:30]:30s}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-omnispatial-val", action="store_true",
                        help="reuse an existing omnispatial val parquet (debug)")
    args = parser.parse_args()

    if not args.skip_omnispatial_val:
        build_omnispatial_val()
    else:
        assert OUT_OMNISPATIAL_VAL.exists(), f"missing {OUT_OMNISPATIAL_VAL}"

    train_n = concat_parquets(
        [SRC_NEW_TRAIN_V2, "/storage/openpsi/data/reasonmap_rl/combined_train.parquet",
         SRC_OMNISPATIAL_TRAIN_VT],
        OUT_TRAIN, "TRAIN",
    )
    val_n = concat_parquets(
        [SRC_REASONMAP_VAL_10PCT, SRC_NEW_VAL, OUT_OMNISPATIAL_VAL],
        OUT_VAL, "VAL",
    )

    summarize(OUT_TRAIN, "TRAIN")
    summarize(OUT_VAL, "VAL")

    print("\nDone.")
    print(f"  Use these in run script:")
    print(f"    train_data=\"[{OUT_TRAIN}]\"")
    print(f"    val_data=\"[{OUT_VAL}]\"")


if __name__ == "__main__":
    main()
