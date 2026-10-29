"""Project paths and the small artifact format shared by experiment runners."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value, *, newline=None):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline=newline)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def canonical_jsonl(records):
    """The dataset digest uses UTF-8 JSONL with sorted keys and LF newlines."""
    return ''.join(json.dumps(record, sort_keys=True) + '\n' for record in records)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(names):
    return {name: file_sha256(ROOT / name) for name in names}


def adapter_hashes(directory):
    return {path.name: file_sha256(path) for path in Path(directory).iterdir()
            if path.suffix in ('.json', '.safetensors')}
