from __future__ import annotations

import csv
import hashlib
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

MANIFEST_FIELDS = [
    "uid",
    "dataset",
    "split",
    "source_id",
    "clip",
    "transcript_text",
    "anchor_text",
    "motion_path",
    "motion_archive_id",
]

# Text sources embedded by `preprocess`. Each maps to a column of the BEAT2 anchor file;
# TED rows use the gesture-lexicon description for every anchor source.
TEXT_SOURCES = {
    "transcript": None,
    "anchor": "anchor",
    "physical_motion_llm": "physical_form",
    "semantic_intent_llm": "communicative_intent",
}


def sanitize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(path: str | Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def normalize_clip_name(value: str) -> str:
    """Map a clip file name (`...---w5s.npz`) or clip id to the clip id used by the anchor file."""
    stem = Path(sanitize_text(value)).stem.strip().strip('"')
    if stem.endswith("---w5s"):
        stem = stem[: -len("---w5s")]
    return stem


def _build_beat2_rows(cfg: dict[str, Any]) -> list[dict[str, str]]:
    paths = cfg["paths"]
    anchors_by_clip: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(paths["beat2_anchors"]):
        clip = normalize_clip_name(row.get("clip", ""))
        if not clip or not sanitize_text(row.get("anchor")) or not sanitize_text(row.get("anchor_id")):
            continue
        anchors_by_clip.setdefault(clip, row)

    clip_dir = Path(paths["beat2_metadata"]).parent
    rows: list[dict[str, str]] = []
    for meta in read_csv_rows(paths["beat2_metadata"]):
        clip = normalize_clip_name(meta.get("output_filename", ""))
        anchor_row = anchors_by_clip.get(clip)
        if anchor_row is None:
            continue
        motion_path = clip_dir / sanitize_text(meta.get("output_filename"))
        if not motion_path.exists():
            continue
        transcript = sanitize_text(meta.get("transcript_windowed"))
        if not transcript:
            continue
        anchor_id = sanitize_text(anchor_row["anchor_id"])
        rows.append(
            {
                "uid": f"beat:{anchor_id}",
                "dataset": "beat",
                "split": "",
                "source_id": anchor_id,
                "clip": clip,
                "transcript_text": transcript,
                "anchor_text": sanitize_text(anchor_row["anchor"]),
                "motion_path": str(motion_path.resolve()),
                "motion_archive_id": "",
            }
        )

    split_cfg = cfg["data"]["split"]
    rows.sort(key=lambda row: row["uid"])
    random.Random(int(split_cfg["seed"])).shuffle(rows)
    num_train = int(len(rows) * float(split_cfg["train_ratio"]))
    num_val = int(len(rows) * float(split_cfg["val_ratio"]))
    for index, row in enumerate(rows):
        if index < num_train:
            row["split"] = "beat_train"
        elif index < num_train + num_val:
            row["split"] = "beat_val"
        else:
            row["split"] = "beat_test"
    _check_published_split(rows, anchors_by_clip)
    return sorted(rows, key=lambda row: row["uid"])


def _check_published_split(rows: list[dict[str, str]], anchors_by_clip: dict[str, dict[str, str]]) -> None:
    """Fail if clips are missing or the split differs from the `split` column of the anchor file."""
    published = {
        f"beat:{sanitize_text(row['anchor_id'])}": f"beat_{sanitize_text(row.get('split'))}"
        for row in anchors_by_clip.values()
        if sanitize_text(row.get("split"))
    }
    if not published:
        return
    built = {row["uid"]: row["split"] for row in rows}
    missing = sorted(set(published) - set(built))
    if missing:
        raise ValueError(
            f"{len(missing)} anchored BEAT2 gestures have no clip or transcript (e.g. {missing[:3]}); "
            "run scripts/prepare_beat2.py on the complete BEAT2 English v2.0.0 data."
        )
    mismatched = [uid for uid, split in built.items() if published.get(uid) != split]
    if mismatched:
        raise ValueError(f"{len(mismatched)} clips fall into a different split than published (e.g. {mismatched[:3]}).")


def _build_ted_rows(cfg: dict[str, Any]) -> list[dict[str, str]]:
    paths = cfg["paths"]
    ted_csv = sanitize_text(paths.get("ted_eval"))
    if not ted_csv or not Path(ted_csv).exists():
        return []
    has_motion = bool(sanitize_text(paths.get("ted_motion"))) and Path(paths["ted_motion"]).exists()
    rows: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(ted_csv):
        sample_id = sanitize_text(row.get("sample_id"))
        query = sanitize_text(row.get("query_text"))
        description = sanitize_text(row.get("lexicon_description"))
        if not sample_id or not query or not description:
            continue
        uid = f"ted:{sample_id}"
        rows.setdefault(
            uid,
            {
                "uid": uid,
                "dataset": "ted",
                "split": "ted_eval",
                "source_id": sample_id,
                "clip": sanitize_text(row.get("id")),
                "transcript_text": query,
                "anchor_text": description,
                "motion_path": "",
                "motion_archive_id": sample_id if has_motion else "",
            },
        )
    return sorted(rows.values(), key=lambda row: row["uid"])


def build_manifest_rows(cfg: dict[str, Any]) -> list[dict[str, str]]:
    return _build_beat2_rows(cfg) + _build_ted_rows(cfg)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    return read_csv_rows(path)


def source_texts(cfg: dict[str, Any], manifest_rows: Sequence[dict[str, str]], source: str) -> dict[str, str]:
    """Return {uid: text} for one text source; rows without text for that source are skipped."""
    if source == "transcript":
        return {row["uid"]: row["transcript_text"] for row in manifest_rows}
    if source == "anchor":
        return {row["uid"]: row["anchor_text"] for row in manifest_rows}
    column = TEXT_SOURCES[source]
    beat_texts = {
        f"beat:{sanitize_text(row.get('anchor_id'))}": sanitize_text(row.get(column))
        for row in read_csv_rows(cfg["paths"]["beat2_anchors"])
    }
    texts: dict[str, str] = {}
    for row in manifest_rows:
        text = beat_texts.get(row["uid"], "") if row["dataset"] == "beat" else row["anchor_text"]
        if text:
            texts[row["uid"]] = text
    return texts


class TextEmbeddingStore:
    """Frozen text-encoder embeddings for one text source, keyed by manifest uid."""

    def __init__(self, npz_path: str | Path) -> None:
        payload = np.load(npz_path, allow_pickle=False)
        uids = payload["uid"].astype(str).tolist()
        embeddings = payload["embedding"].astype(np.float32)
        self.lookup = {uid: embeddings[index] for index, uid in enumerate(uids)}
        self.dim = int(embeddings.shape[-1])

    def stack(self, uids: Sequence[str]) -> torch.Tensor:
        return torch.from_numpy(np.stack([self.lookup[uid] for uid in uids]))


def build_fixed_uid_derangement(uids: Sequence[str], seed: int, group_key: str) -> dict[str, str]:
    """Map every uid to a different uid with one deterministic cyclic shift."""
    uid_list = list(uids)
    if len(uid_list) <= 1:
        return {uid: uid for uid in uid_list}
    digest = hashlib.sha256(f"{seed}::{group_key}".encode("utf-8")).digest()
    offset = 1 + (int.from_bytes(digest[:8], "little") % (len(uid_list) - 1))
    shifted = uid_list[offset:] + uid_list[:offset]
    return dict(zip(uid_list, shifted))


@lru_cache(maxsize=256)
def _load_clip_motion(path: str) -> np.ndarray:
    poses = np.asarray(np.load(path, allow_pickle=True)["poses"])
    return np.nan_to_num(poses.reshape(poses.shape[0], -1).astype(np.float32, copy=False))


@lru_cache(maxsize=2)
def _load_motion_archive(path: str) -> tuple[dict[int, int], np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    index = {int(sample_id): position for position, sample_id in enumerate(payload["sample_id"].tolist())}
    return index, payload["data_poses"]


def _load_archive_motion(path: str, sample_id: int) -> np.ndarray:
    index, sequences = _load_motion_archive(path)
    poses = np.asarray(sequences[index[sample_id]])
    return np.nan_to_num(poses.reshape(poses.shape[0], -1).astype(np.float32, copy=False))


class GestureDataset(Dataset):
    """Manifest rows with their transcript/anchor embeddings and (optionally) motion.

    `extra_stores` hold the auxiliary anchor sources (physical form, intent); only rows that
    have an embedding for every extra source are kept. `mismatched_source_groups` pairs each
    listed source with another sample's text through a fixed derangement (mismatched-anchor
    control); sources in the same group share one derangement.
    """

    def __init__(
        self,
        manifest_rows: Sequence[dict[str, str]],
        transcript_store: TextEmbeddingStore,
        anchor_store: TextEmbeddingStore,
        splits: str | Iterable[str],
        include_motion: bool,
        extra_stores: dict[str, TextEmbeddingStore] | None = None,
        ted_motion_path: str = "",
        mismatched_source_groups: Sequence[Sequence[str]] = (),
        mismatch_seed: int = 42,
    ) -> None:
        split_set = {splits} if isinstance(splits, str) else set(splits)
        self.transcript_store = transcript_store
        self.anchor_store = anchor_store
        self.extra_stores = dict(extra_stores or {})
        self.include_motion = include_motion
        self.ted_motion_path = ted_motion_path
        self.rows = [row for row in manifest_rows if row["split"] in split_set]
        if self.extra_stores:
            allowed = set(anchor_store.lookup)
            for store in self.extra_stores.values():
                allowed &= set(store.lookup)
            self.rows = [row for row in self.rows if row["uid"] in allowed]

        self.donor_maps: dict[str, dict[str, str]] = {}
        row_uids = [row["uid"] for row in self.rows]
        singleton_maps: list[dict[str, str]] = []
        for group_index, group in enumerate(mismatched_source_groups):
            attempt = 0
            while True:
                mapping = build_fixed_uid_derangement(
                    row_uids,
                    seed=mismatch_seed,
                    group_key=(
                        f"{group_index}::{'::'.join(group)}::"
                        f"{'::'.join(sorted(split_set))}::attempt{attempt}"
                    ),
                )
                # Independently mismatched sources must never share a donor for any sample.
                if len(group) != 1 or not any(
                    any(mapping[uid] == previous[uid] for uid in row_uids) for previous in singleton_maps
                ):
                    break
                attempt += 1
            for source in group:
                self.donor_maps[source] = mapping
            if len(group) == 1:
                singleton_maps.append(mapping)

    def __len__(self) -> int:
        return len(self.rows)

    def _motion(self, row: dict[str, str]) -> np.ndarray:
        if row["motion_path"]:
            return _load_clip_motion(row["motion_path"])
        if not row["motion_archive_id"]:
            raise ValueError(f"No motion available for {row['uid']}.")
        return _load_archive_motion(self.ted_motion_path, int(row["motion_archive_id"]))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        uid = row["uid"]
        item: dict[str, Any] = {
            "uid": uid,
            "clip": row["clip"],
            "anchor_embedding": torch.tensor(self.anchor_store.lookup[uid]),
            "transcript_embedding": torch.tensor(self.transcript_store.lookup[uid]),
        }
        if self.extra_stores:
            item["extra_anchor_embeddings"] = {
                source: torch.tensor(store.lookup[self.donor_maps.get(source, {}).get(uid, uid)])
                for source, store in self.extra_stores.items()
            }
        if self.include_motion:
            item["motion"] = torch.tensor(self._motion(row))
        return item


def collate_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "uids": [item["uid"] for item in batch],
        "clips": [item["clip"] for item in batch],
        "anchor_embeddings": torch.stack([item["anchor_embedding"] for item in batch]),
        "transcript_embeddings": torch.stack([item["transcript_embedding"] for item in batch]),
    }
    if "extra_anchor_embeddings" in batch[0]:
        result["extra_anchor_embeddings"] = {
            source: torch.stack([item["extra_anchor_embeddings"][source] for item in batch])
            for source in batch[0]["extra_anchor_embeddings"]
        }
    if "motion" in batch[0]:
        lengths = [item["motion"].shape[0] for item in batch]
        motions = torch.zeros((len(batch), max(lengths), batch[0]["motion"].shape[-1]), dtype=torch.float32)
        mask = torch.zeros((len(batch), max(lengths)), dtype=torch.bool)
        for index, item in enumerate(batch):
            motions[index, : lengths[index]] = item["motion"]
            mask[index, : lengths[index]] = True
        result["motions"] = motions
        result["motion_mask"] = mask
    return result
