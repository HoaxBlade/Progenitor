# Phase 2: Software

Artifacts are directories containing a **progenitor.yaml** manifest. Progenitor only applies levers you explicitly enable (no auto-blast).

- **CLI:** `progenitor enhance-software <artifact_dir> [--tune-workers] [--workers N]` — writes `.env.progenitor` in the artifact dir (or `--output-env`).
- **Benchmark:** `python benchmarks/run_software.py --url <base_url> --repeat 50` — you start the server before/after; script reports latency and throughput.
