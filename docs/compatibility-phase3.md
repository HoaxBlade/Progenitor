# Phase 3: Progenitor compatibility spec (Devices)

Progenitor Phase 3 applies “enhance to peak” to **devices**: phones, PCs, and later drones, servers, vehicles, IoT. Input = device (on your network) + target metrics → output = same device, measurably improved performance (target 50×+ where achievable).

**Status:** Phase 3 started; spec and first targets defined incrementally.

---

## Goal

- **Reach** a device on your network (customer has asked you to enhance it; you control the flow).
- **Measure** baseline (throughput, latency, battery, CPU, mission-relevant KPIs).
- **Apply** Progenitor tuning (OS, config, resources — device-type specific).
- **Measure again** and report before/after. Same device, peak performance.

---

## Scope (Phase 3)

### In scope (this repo)

- **Compatibility spec:** This document; which device types and OS versions we support, and where the external access module plugs in.
- **Enhancement pipeline:** Measure baseline → apply enhancements → measure again. Interfaces so an external access module can deploy and run the payload on the device.
- **Operator CLI:** Discover or select a device (e.g. by IP), trigger enhancement (calling the access module), run the pipeline, report metrics.
- **Docs:** How to integrate the access module, how to add new device types.

### Out of scope (this repo)

- **Access mechanism:** The code that establishes a foothold on the device over the network (step 3 of the implementation plan). Implemented and versioned elsewhere.
- **Access module:** The component that runs on your infra, discovers/targets the device, and invokes the access mechanism (step 4). Implemented and integrated elsewhere.

### First supported device types

- **Phones:** Android (versions TBD, e.g. 12+), specific chipsets to be documented.
- **PCs:** Windows 10/11, Linux (x86_64/ARM64). Service/runtime to be documented per OS.

More device types (drones, servers, vehicles, IoT) later. Vision remains “any device.”

---

## Consent and control

- Progenitor runs on a device only when the **user has explicitly asked** to enhance it (e.g. “enhance this phone”). That request is the permission.
- **Same-network, you control:** Customer is on your network; you initiate and run enhancement from your side. No welcome page, no link, no tap from the customer.
- **No customer control flow:** Operator runs the tool and triggers “enhance”; the access module and payload run under your control.

---

## Implementation plan (summary)

| Step | Description | In this repo? |
|------|-------------|----------------|
| 1. Target selection | Device types and OS versions (e.g. Android 12, Windows 10/11). | Yes — compatibility spec. |
| 2. Reachable surface | Per target: which components accept traffic from the LAN. | Yes — document in spec. |
| 3. Access mechanism | Mechanism to establish a foothold over the network. | **No** — elsewhere. |
| 4. Access module | Runs on your infra; discovers device, invokes access, confirms. | **No** — elsewhere. |
| 5. Payload delivery | Deploy and run enhancement (measure → tune → measure). | Yes — enhancement pipeline. |
| 6. Operator flow | CLI: select device → run access module → run pipeline → report. | Yes — operator CLI. |

---

## Compatibility: when is a device “Progenitor-compatible”?

A device is **Progenitor-compatible for Phase 3** if:

1. **Target type** is one of the supported device types (e.g. phone, PC) and OS/version is in the supported set for that type.
2. **Reachable:** The access module (external) can reach it on the network (e.g. by IP/host) and establish a session.
3. **Measurable:** We can run the enhancement payload on the device (or via the access channel) and collect baseline and after metrics (throughput, latency, CPU, battery, or mission KPIs).
4. **Tunable:** We have defined, reversible enhancements for that device type (config, OS settings, resource allocation). Changes are documented and rollback is possible.

**First targets (to refine):**

- **Android 12+** (phones): Reachable surface and access mechanism TBD; documented when available. Enhancement levers: e.g. battery/performance profile, background limits, thermal policy.
- **Windows 10/11** (PCs): Service or management interface on LAN; enhancement levers: power plan, CPU affinity, background apps, visual effects.
- **Linux** (PCs/servers): SSH or agent; enhancement levers: governor, CPU affinity, I/O scheduler, kernel params.

---

## Enhancement pipeline (in-repo)

The pipeline is **measure → tune → measure**:

1. **Measure baseline:** Run a small suite of benchmarks on the device (or via the access channel). Collect metrics (e.g. CPU score, I/O throughput, latency, battery drain rate). Result: `DeviceBaseline` (or similar).
2. **Apply enhancements:** For the device type, apply opt-in tuning (e.g. power profile, governor, disabled animations). All changes are explicit and reversible; no “improve everything” by default.
3. **Measure after:** Re-run the same suite; collect `DeviceAfter` metrics.
4. **Report:** Before/after comparison, speedup ratio(s), and list of applied changes.

The **access module** (external) is responsible for:

- Discovering or accepting a device identifier (e.g. IP, hostname, device ID).
- Establishing a session (using the access mechanism, not in this repo).
- **Executing the payload:** This repo provides the payload (scripts/configs + orchestration). The access module runs them on the device and returns stdout/stderr and result files (e.g. metrics JSON).

**Plug point:** The operator CLI calls an **access adapter** (interface). Default: local/mock adapter for development. Production: adapter that invokes the external access module (e.g. subprocess, or library API if the module is linked). Adapter interface: `establish(device_id) -> DeviceSession`, and `DeviceSession.run_payload(payload_dir_or_cmd) -> PayloadResult`.

---

## Operator CLI (in-repo)

- **Command:** `progenitor enhance-device [--device DEVICE] [--list] [--dry-run]`
  - `--device`: Device identifier (IP, hostname, or ID). If omitted and not `--list`, prompt or use default from env (e.g. `PROGENITOR_DEVICE`).
  - `--list`: Discover devices on the LAN (delegate to access module if available; otherwise show message).
  - `--dry-run`: Run measure → tune → measure locally or in mock mode (no real device).
- **Flow:** Resolve adapter → establish session (or use mock) → run enhancement pipeline → print before/after and applied changes.

---

## Metrics (device types)

| Device type | Example baseline/after metrics |
|-------------|--------------------------------|
| Phone       | CPU score, battery drain (mA or %/hr), frame rate, cold start time |
| PC (Windows/Linux) | CPU score, disk throughput, idle/load power, boot time |

Exact metrics and units per target will be defined in the enhancement pipeline and compatibility table.

---

## Version

- **Spec version:** 0.1  
- **Phase:** 3 (Devices)  
- **Last updated:** 2026-03  
