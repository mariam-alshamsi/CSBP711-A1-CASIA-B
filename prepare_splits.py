from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SEED = 42

df = pd.read_csv(OUT / "data_manifest.csv")

# Preserve an explicit record of excluded sequences.
excluded = df[df["frames"] < 16].copy()
excluded["exclusion_reason"] = "Fewer than 16 frames"
excluded.to_csv(OUT / "excluded_sequences.csv", index=False)

kept = df[df["frames"] >= 16].copy()

# Choose validation subjects only from the development identities.
development_ids = np.sort(
    kept.loc[kept["split"] == "development", "subject"].unique()
)
rng = np.random.default_rng(SEED)
validation_ids = sorted(
    int(x) for x in rng.choice(development_ids, size=14, replace=False)
)

kept["role"] = kept["split"]

is_development = kept["split"] == "development"
is_validation = kept["subject"].isin(validation_ids)

kept.loc[is_development & ~is_validation, "role"] = "train"

# Validation mirrors the final gallery–probe task.
gallery_sequences = {"nm-01", "nm-02", "nm-03", "nm-04"}
is_gallery_sequence = kept["sequence"].isin(gallery_sequences)

kept.loc[
    is_development & is_validation & is_gallery_sequence,
    "role"
] = "validation_gallery"

kept.loc[
    is_development & is_validation & ~is_gallery_sequence,
    "role"
] = "validation_probe"

kept.loc[kept["split"] == "gallery", "role"] = "test_gallery"
kept.loc[kept["split"] == "probe", "role"] = "test_probe"

# Verify that representation learning cannot see validation/test identities.
train_ids = set(kept.loc[kept["role"] == "train", "subject"])
val_ids = set(validation_ids)
test_ids = set(
    kept.loc[kept["role"].str.startswith("test_"), "subject"]
)

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

for prefix in ("validation", "test"):
    gallery_ids = set(
        kept.loc[kept["role"] == f"{prefix}_gallery", "subject"]
    )
    probe_ids = set(
        kept.loc[kept["role"] == f"{prefix}_probe", "subject"]
    )
    assert probe_ids <= gallery_ids, "Probe identity missing from gallery"

kept.to_csv(OUT / "split_manifest.csv", index=False)

config = {
    "seed": SEED,
    "minimum_frames": 16,
    "training_subjects": sorted(int(x) for x in train_ids),
    "validation_subjects": validation_ids,
    "test_subjects": sorted(int(x) for x in test_ids),
}

with (OUT / "split_config.json").open("w") as handle:
    json.dump(config, handle, indent=2)

print("Excluded sequences:", len(excluded))
print("Retained sequences:", len(kept))
print("\nSequences by role:")
print(kept["role"].value_counts().sort_index().to_string())
print("\nTraining subjects:", len(train_ids))
print("Validation subjects:", len(val_ids))
print("Test subjects:", len(test_ids))
print("\nValidation subject IDs:", validation_ids)
print("\nSubject separation checks passed.")