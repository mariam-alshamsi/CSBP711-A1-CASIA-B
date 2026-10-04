from pathlib import Path
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
manifest = pd.read_csv(OUT / "split_manifest.csv")

# Use validation data while developing the experiment.
gallery_rows = manifest[
    manifest["role"] == "validation_gallery"
].reset_index(drop=True)

probe_rows = manifest[
    manifest["role"] == "validation_probe"
].reset_index(drop=True)

CONDITIONS = {
    "L-4": "Clean",
    "L-0": "Bottom 25%",
    "L-1": "Bottom 50%",
    "L-2": "Left 50%",
    "L-3": "Right 50%",
}


def load_gei(relative_path):
    with (ROOT / relative_path).open("rb") as handle:
        frames = pickle.load(handle)

    frames = frames.astype(np.float32)

    # Fixed normalization, rather than scaling each sequence separately.
    if frames.min() < 0 or frames.max() > 255:
        raise ValueError(f"Unexpected pixel range: {relative_path}")

    frames /= 255.0
    return frames.mean(axis=0)


def apply_mask(image, label):
    result = image.copy()
    height, width = result.shape

    if label == "L-0":
        result[3 * height // 4:, :] = 0
    elif label == "L-1":
        result[height // 2:, :] = 0
    elif label == "L-2":
        result[:, :width // 2] = 0
    elif label == "L-3":
        result[:, width // 2:] = 0

    return result


print("Loading validation gallery and probes...")
start = time.perf_counter()
gallery = np.stack([load_gei(p) for p in gallery_rows["path"]])
probes = np.stack([load_gei(p) for p in probe_rows["path"]])
feature_seconds = time.perf_counter() - start

gallery_vectors = gallery.reshape(len(gallery), -1)
gallery_ids = gallery_rows["subject"].to_numpy()
probe_ids = probe_rows["subject"].to_numpy()

# This baseline has no learned parameters.
# Its reference gallery is the state required for recognition.
reference_path = OUT / "gei_validation_gallery.npz"
np.savez_compressed(
    reference_path,
    features=gallery_vectors,
    subject_ids=gallery_ids,
)
reference_bytes = reference_path.stat().st_size

# Save a visual check of the mask definitions.
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for ax, (label, name) in zip(axes, CONDITIONS.items()):
    ax.imshow(
        apply_mask(probes[0], label),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"{label}\n{name}")
    ax.axis("off")

fig.suptitle(
    f"Validation GEI example — subject {probe_ids[0]:03d}"
)
fig.tight_layout()
fig.savefig(OUT / "occlusion_examples.png", dpi=180)
plt.close(fig)

results = []
predictions = []

for label in CONDITIONS:
    masked = np.stack([apply_mask(image, label) for image in probes])
    probe_vectors = masked.reshape(len(masked), -1)

    start = time.perf_counter()

    # Squared Euclidean distance has the same nearest-neighbour
    # ranking as Euclidean distance.
    if not np.isfinite(probe_vectors).all():
        raise ValueError("Probe features contain NaN or infinity.")
    if not np.isfinite(gallery_vectors).all():
        raise ValueError("Gallery features contain NaN or infinity.")

    distances = cdist(
        probe_vectors.astype(np.float64),
        gallery_vectors.astype(np.float64),
        metric="sqeuclidean",
    )

    if not np.isfinite(distances).all():
        raise ValueError("Distances contain NaN or infinity.")

    predicted = gallery_ids[np.argmin(distances, axis=1)]
    matching_seconds = time.perf_counter() - start
    correct = predicted == probe_ids

    for condition in ("nm", "bg", "cl"):
        selected = probe_rows["condition"].to_numpy() == condition
        results.append({
            "method": "GEI nearest-gallery",
            "evaluation": "validation",
            "occlusion": label,
            "walking_condition": condition,
            "n_probes": int(selected.sum()),
            "rank1_accuracy_percent": float(
                100 * correct[selected].mean()
            ),
            "training_seconds": 0.0,
            "feature_extraction_seconds_all": feature_seconds,
            "matching_seconds_all": matching_seconds,
            "gallery_storage_bytes": reference_bytes,
        })

    for index, row in probe_rows.iterrows():
        predictions.append({
            "path": row["path"],
            "subject": int(probe_ids[index]),
            "condition": row["condition"],
            "occlusion": label,
            "predicted_subject": int(predicted[index]),
            "correct": bool(correct[index]),
        })

result_df = pd.DataFrame(results)
result_df.to_csv(OUT / "gei_validation_results.csv", index=False)
pd.DataFrame(predictions).to_csv(
    OUT / "gei_validation_predictions.csv", index=False
)

table = result_df.pivot(
    index="occlusion",
    columns="walking_condition",
    values="rank1_accuracy_percent",
).reindex(CONDITIONS.keys())

print("\nVALIDATION RANK-1 ACCURACY (%)")
print(table.round(2).to_string())
print(f"\nFeature extraction: {feature_seconds:.2f} seconds")
print(f"Stored gallery: {reference_bytes / 1024:.1f} KiB")
print("Training: none — this is a direct-matching baseline.")
print("\nSaved results and occlusion_examples.png in results/")