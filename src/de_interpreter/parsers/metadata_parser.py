"""Parser for experimental metadata."""

from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import json
import yaml


@dataclass
class ExperimentalContext:
    """Container for experimental context information."""

    disease: str
    tissue: Optional[str] = None
    cell_type: Optional[str] = None
    treatment: Optional[str] = None
    control: Optional[str] = None
    time_point: Optional[str] = None
    organism: str = "human"
    comparison_description: Optional[str] = None
    sample_size: Optional[Dict[str, int]] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def get_search_terms(self) -> List[str]:
        """Generate search terms for literature queries."""
        terms = [self.disease]

        if self.tissue:
            terms.append(self.tissue)
        if self.cell_type:
            terms.append(self.cell_type)
        if self.treatment:
            terms.append(self.treatment)

        return terms

    def get_context_string(self) -> str:
        """Generate a human-readable context description."""
        parts = []

        if self.comparison_description:
            parts.append(self.comparison_description)
        else:
            if self.treatment and self.control:
                parts.append(f"{self.treatment} vs {self.control}")

            parts.append(f"in {self.disease}")

            if self.tissue:
                parts.append(f"({self.tissue}")
                if self.cell_type:
                    parts.append(f"{self.cell_type}")
                parts.append(")")

            if self.time_point:
                parts.append(f"at {self.time_point}")

        return " ".join(parts)


class MetadataParser:
    """Parser for experimental metadata files."""

    REQUIRED_FIELDS = ["disease"]

    def __init__(self):
        self.metadata: Optional[Dict[str, Any]] = None
        self.context: Optional[ExperimentalContext] = None

    def parse(self, file_path: Path) -> ExperimentalContext:
        """Parse metadata from file."""
        self.metadata = self._read_file(file_path)
        self._validate_metadata()
        self.context = self._create_context()
        return self.context

    def parse_dict(self, metadata: Dict[str, Any]) -> ExperimentalContext:
        """Parse metadata from dictionary."""
        self.metadata = metadata
        self._validate_metadata()
        self.context = self._create_context()
        return self.context

    def _read_file(self, file_path: Path) -> Dict[str, Any]:
        """Read metadata from JSON or YAML file."""
        suffix = file_path.suffix.lower()

        with open(file_path, "r") as f:
            if suffix == ".json":
                return json.load(f)
            elif suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported metadata format: {suffix}")

    def _validate_metadata(self) -> None:
        """Validate required fields are present."""
        if not self.metadata:
            raise ValueError("Empty metadata")

        missing = [f for f in self.REQUIRED_FIELDS if f not in self.metadata]
        if missing:
            raise ValueError(f"Missing required metadata fields: {missing}")

    def _create_context(self) -> ExperimentalContext:
        """Create ExperimentalContext from metadata."""
        # Extract standard fields
        context_args = {
            "disease": self.metadata["disease"],
            "tissue": self.metadata.get("tissue"),
            "cell_type": self.metadata.get("cell_type"),
            "treatment": self.metadata.get("treatment"),
            "control": self.metadata.get("control"),
            "time_point": self.metadata.get("time_point"),
            "organism": self.metadata.get("organism", "human"),
            "comparison_description": self.metadata.get("comparison_description"),
            "sample_size": self.metadata.get("sample_size"),
        }

        # Store any additional fields
        standard_fields = set(context_args.keys())
        additional = {
            k: v for k, v in self.metadata.items() if k not in standard_fields
        }

        context_args["additional_info"] = additional

        return ExperimentalContext(**context_args)
