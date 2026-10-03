"""Typed report container shared by model training workflows."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StandardReport:
    """Mutable report with a uniform run and data metadata contract."""

    report_version: str
    run: dict[str, Any]
    data: dict[str, Any]
    sections: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key in {"report_version", "run", "data"}:
            return self.__dict__[key]
        else:
            return self.sections[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in {"report_version", "run", "data"}:
            self.__dict__[key] = value
        else:
            self.sections[key] = value

    def __contains__(self, key: object) -> bool:
        return key in {"report_version", "run", "data"} or key in self.sections

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable report payload."""
        return {
            "report_version": self.report_version,
            "run": self.run,
            "data": self.data,
            **self.sections,
        }

    def attach_data_manifest(
        self,
        manifest: dict[str, Any],
        local_path: str,
        s3_uri: str,
    ) -> None:
        """Attach all data-lineage references under the standard data section."""
        self.data.update(
            {
                "data_version_id": manifest["data_version_id"],
                "manifest_local_path": local_path,
                "manifest_s3_uri": s3_uri,
                "raw_inventory": manifest.get("raw_inventory"),
            }
        )