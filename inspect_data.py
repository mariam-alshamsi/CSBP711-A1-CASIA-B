from pathlib import Path
from collections import Counter
import hashlib
import json
import pickle

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "casia-b"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

files = sorted(DATA.glob("*/*/090/*.pkl"))
if not files:
    raise SystemExit(f"No 90-degree sequence files found in {DATA}")

records = []
errors = []

for path in files:
    subject, sequence, view, _ = path.relative_to(DATA).parts
    subject_id = int(subject)

    if subject_id <= 74:
        split = "development"
    elif sequence in {"nm-01", "nm-02", "nm-03", "nm-04"}:
        split = "gallery"
    else:
        split = "probe"

    try:
        with path.open("rb") as handle:
            frames = pickle.load(handle)

        if not isinstance(frames, np.ndarray):
            raise ValueError("Expected a NumPy array")
        if frames.ndim != 3 or frames.shape[1:] != (64, 64):
            raise ValueError(f"Unexpected shape: {frames.shape}")
        if len(frames) == 0:
            raise ValueError("Empty sequence")
        if not np.isfinite(frames).all():
            raise ValueError("Non-finite pixel values")

        # Include shape and dtype when checking identical sequence content.
        digest = hashlib.sha256()
        digest.update(str(frames.shape).encode())
        digest.update(str(frames.dtype).encode())
        digest.update(frames.tobytes())

        records.append({
            "path": path.relative_to(ROOT).as_posix(),
            "subject": subject_id,
            "sequence": sequence,
            "condition": sequence[:2],
            "view": view,
            "split": split,
            "frames": len(frames),
            "height": frames.shape[1],
            "width": frames.shape[2],
            "dtype": str(frames.dtype),
            "pixel_min": float(frames.min()),
            "pixel_max": float(frames.max()),
            "empty_frames": int(
                np.count_nonzero(
                    np.count_nonzero(frames, axis=(1, 2)) == 0
                )
            ),
            "under_16_frames": len(frames) < 16,
            "sha256": digest.hexdigest(),
        })

    except Exception as exc:
        errors.append({
            "path": path.relative_to(ROOT).as_posix(),
            "error": str(exc),
        })

    if (len(records) + len(errors)) % 100 == 0:
        print(f"Checked {len(records) + len(errors)}/{len(files)}")

df = pd.DataFrame(records)
pd.DataFrame(errors, columns=["path", "error"]).to_csv(
    OUT / "data_errors.csv", index=False
)

if df.empty:
    raise SystemExit("No valid sequences. See results/data_errors.csv")

df.to_csv(OUT / "data_manifest.csv", index=False)

duplicates = df[df.duplicated("sha256", keep=False)]
duplicates.to_csv(OUT / "duplicate_sequences.csv", index=False)

# Check the expected 10 recordings for every subject.
expected_sequences = (
    [f"nm-{i:02d}" for i in range(1, 7)]
    + ["bg-01", "bg-02", "cl-01", "cl-02"]
)
observed = Counter(
    (row["subject"], row["sequence"]) for row in records
)

missing = [
    {"subject": subject, "sequence": sequence}
    for subject in range(1, 125)
    for sequence in expected_sequences
    if observed[(subject, sequence)] == 0
]
pd.DataFrame(missing, columns=["subject", "sequence"]).to_csv(
    OUT / "missing_sequences.csv", index=False
)

cross_split_duplicates = int(
    (df.groupby("sha256")["split"].nunique() > 1).sum()
)

summary = {
    "files_found": len(files),
    "valid_sequences": len(df),
    "unreadable_or_invalid": len(errors),
    "subjects": int(df["subject"].nunique()),
    "total_frames": int(df["frames"].sum()),
    "minimum_frames": int(df["frames"].min()),
    "maximum_frames": int(df["frames"].max()),
    "sequences_under_16_frames": int(df["under_16_frames"].sum()),
    "empty_frames": int(df["empty_frames"].sum()),
    "missing_expected_sequences": len(missing),
    "duplicate_content_groups": int(duplicates["sha256"].nunique()),
    "duplicate_groups_crossing_splits": cross_split_duplicates,
    "sequences_by_split": {
        key: int(value)
        for key, value in df["split"].value_counts().items()
    },
    "sequences_by_condition": {
        key: int(value)
        for key, value in df["condition"].value_counts().items()
    },
}

with (OUT / "data_summary.json").open("w") as handle:
    json.dump(summary, handle, indent=2)

print("\nDATA AUDIT")
print(json.dumps(summary, indent=2))
print("\nReports saved in:", OUT)