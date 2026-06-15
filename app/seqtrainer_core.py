from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdflib import Graph
from rdflib.query import ResultRow
from sklearn.preprocessing import OneHotEncoder


DNA_BASES = ["A", "C", "G", "T"]
DNA_WITH_N = ["A", "C", "G", "T", "N"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sequence(seq: str) -> str:
    return "".join(base if base in DNA_WITH_N else "N" for base in str(seq).upper().strip())


def pad_sequence(seq: str, max_length: int) -> str:
    seq = normalize_sequence(seq)
    if len(seq) > max_length:
        diff = len(seq) - max_length
        trim_length = int(diff / 2)
        return seq[trim_length : -(trim_length + diff % 2)]
    return seq.center(max_length, "N")


def one_hot_encode(sequences: Iterable[str], defined_categories: list[str] | None = None) -> np.ndarray:
    categories = defined_categories or DNA_WITH_N
    sequence_rows = np.array([[char for char in seq] for seq in sequences])
    sequence_length = sequence_rows.shape[1]
    encoder = OneHotEncoder(
        sparse_output=False,
        categories=[categories] * sequence_length,
        dtype=np.float32,
    )
    return encoder.fit_transform(sequence_rows)


def calc_gc(df: pd.DataFrame, seq_col_name: str) -> pd.DataFrame:
    values = []
    for seq in df[seq_col_name]:
        normalized = normalize_sequence(seq)
        length = len(normalized)
        values.append((normalized.count("G") + normalized.count("C")) / length if length else 0)
    return pd.DataFrame({"gc_content": values})


def generate_kmer_counts(df: pd.DataFrame, seq_col_name: str, k: int, normalize: bool = True) -> pd.DataFrame:
    all_kmers = ["".join(kmer) for kmer in itertools.product(DNA_BASES, repeat=k)]
    rows = []
    for seq in df[seq_col_name]:
        normalized = normalize_sequence(seq)
        counts = Counter(normalized[i : i + k] for i in range(len(normalized) - k + 1))
        total = max(len(normalized) - k + 1, 1)
        rows.append({kmer: counts.get(kmer, 0) / total if normalize else counts.get(kmer, 0) for kmer in all_kmers})
    return pd.DataFrame(rows, columns=all_kmers)


def read_tabular_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    separator = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def read_fasta_dataset(path: Path) -> pd.DataFrame:
    records = []
    current_id: str | None = None
    current_parts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records.append({"id": current_id, "sequence": normalize_sequence("".join(current_parts))})
            current_id = line[1:].strip() or f"sequence_{len(records) + 1}"
            current_parts = []
        else:
            current_parts.append(line)
    if current_id is not None:
        records.append({"id": current_id, "sequence": normalize_sequence("".join(current_parts))})
    return pd.DataFrame(records)


def get_sequence_from_sbol(path: Path) -> str | None:
    graph = Graph()
    graph.parse(str(path), format="xml")
    query = """
    PREFIX sbol: <http://sbols.org/v2#>
    SELECT ?sequence
    WHERE { ?s sbol:elements ?sequence . }
    """
    result = graph.query(query)
    for row in result:
        if isinstance(row, ResultRow):
            return normalize_sequence(str(row.sequence))
    return None


def find_numeric_sbol_values(path: Path) -> list[float]:
    graph = Graph()
    graph.parse(str(path), format="xml")
    query = """
    SELECT DISTINCT ?value
    WHERE { ?item ?predicate ?value . }
    """
    values = []
    for row in graph.query(query):
        if isinstance(row, ResultRow):
            try:
                values.append(float(row.value))
            except (TypeError, ValueError):
                continue
    return values


def read_sbol_dataset(path: Path) -> pd.DataFrame:
    sequence = get_sequence_from_sbol(path)
    values = find_numeric_sbol_values(path)
    row = {"id": path.stem, "sequence": sequence or ""}
    if values:
        row["target"] = values[0]
    return pd.DataFrame([row])


def read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        return read_tabular_dataset(path)
    if suffix in {".fa", ".fasta"}:
        return read_fasta_dataset(path)
    if suffix in {".xml", ".rdf", ".sbol"}:
        return read_sbol_dataset(path)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def build_features(df: pd.DataFrame, sequence_col: str, config: dict) -> pd.DataFrame:
    working = pd.DataFrame({"sequence": df[sequence_col].map(normalize_sequence)})
    feature_frames = []

    if config.get("use_gc", True):
        feature_frames.append(calc_gc(working, "sequence"))

    k = int(config.get("kmer_size", 6))
    if config.get("use_kmers", True):
        feature_frames.append(generate_kmer_counts(working, "sequence", k=k, normalize=bool(config.get("normalize_kmers", True))))

    if config.get("use_one_hot", False):
        max_length = int(config.get("sequence_length", 150))
        padded = [pad_sequence(seq, max_length) for seq in working["sequence"]]
        matrix = one_hot_encode(padded)
        feature_frames.append(pd.DataFrame(matrix, columns=[f"ohe_{idx}" for idx in range(matrix.shape[1])]))

    if not feature_frames:
        raise ValueError("Select at least one feature family.")

    return pd.concat(feature_frames, axis=1)
