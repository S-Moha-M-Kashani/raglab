"""Thread-safe in-process preview state for imported experiment archives."""
import copy
import threading
from pathlib import Path

from raglab.evaluation import experiment_archive as archive
from raglab.evaluation import service_experiment_ledger as ledger


class ImportedArchiveStore:
    """Persist completed imports while keeping just one active Inspector view."""

    def __init__(self, path: Path | None = None):
        self._path = path
        self._lock = threading.RLock()
        self._active: dict | None = None

    def import_archive(self, payload: dict) -> dict:
        """Validate and persist before atomically replacing the active view."""
        value = archive.validate_archive(payload)
        if 'evaluation' not in value:
            raise archive.ArchiveError('evaluation: completed archive is required')
        run_id = value['evaluation']['result']['run_id']
        with self._lock:
            disposition = ledger.insert_archive(value, path=self._path)
            self._active = copy.deepcopy(value)
        return {'archive_id': run_id, 'database': disposition}

    def metadata(self) -> dict:
        """Return the small polling payload; the archive itself is fetched on demand."""
        with self._lock:
            run_id = (self._active or {}).get('evaluation', {}).get(
                'result', {}).get('run_id')
        return {'archive_id': run_id, 'source': 'import' if run_id else None}

    def get(self, run_id: str) -> dict | None:
        """Prefer the current preview, otherwise fall back to durable storage."""
        with self._lock:
            active = copy.deepcopy(self._active)
        active_id = (active or {}).get('evaluation', {}).get(
            'result', {}).get('run_id')
        return active if active_id == run_id else ledger.load_archive(
            run_id, path=self._path)

    def clear(self) -> None:
        """Return the Inspector to live data without removing imported history."""
        with self._lock:
            self._active = None
