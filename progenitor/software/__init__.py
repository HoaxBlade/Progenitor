"""Phase 2: Enhance software (services, APIs) to peak performance. User-controlled, opt-in levers only."""

from progenitor.software.manifest import load_manifest
from progenitor.software.enhance import enhance_software, enhance_software_by_url

__all__ = ["load_manifest", "enhance_software", "enhance_software_by_url"]
