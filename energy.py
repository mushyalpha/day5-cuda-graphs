
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


# Throttle reason decoder

# Bitmask constants from nvml.h
_THROTTLE_REASONS = {
    0x0000_0000_0000_0001: "GPU_IDLE",
    0x0000_0000_0000_0002: "APP_CLOCKS_SETTING",
    0x0000_0000_0000_0004: "SW_POWER_CAP",         # ← you'll see this on H100 at 700 W
    0x0000_0000_0000_0008: "HW_SLOWDOWN",
    0x0000_0000_0000_0010: "SYNC_BOOST",
    0x0000_0000_0000_0020: "SW_THERMAL_SLOWDOWN",
    0x0000_0000_0000_0040: "HW_THERMAL_SLOWDOWN",
    0x0000_0000_0000_0080: "HW_POWER_BRAKE",
    0x0000_0000_0000_0100: "DISPLAY_CLOCK_SETTING",
}


def decode_throttle_reasons(bitmask: int) -> list[str]:
    """Decode NVML throttle-reason bitmask into human-readable strings."""
    if bitmask == 0:
        return ["NONE"]
    reasons = []
    for bit, name in _THROTTLE_REASONS.items():
        if bitmask & bit:
            reasons.append(name)
    return reasons or [f"UNKNOWN(0x{bitmask:x})"]


# Cost calculator

def cost_per_million_tokens(
    joules: float,
    n_tokens: int,
    price_per_kwh: float,
) -> float:
    """
    Convert measured energy into £/$/€ per 1M tokens at a given grid price.

    joules:         total energy consumed during measurement
    n_tokens:       total tokens generated
    price_per_kwh:  electricity price (e.g. 0.08 for 8p/kWh, 0.30 for 30p/kWh)

    Returns cost per 1,000,000 tokens in the same currency as price_per_kwh.
    """
    if n_tokens == 0:
        return float("inf")
    kwh = joules / 3_600_000          # 1 kWh = 3.6 MJ
    cost = kwh * price_per_kwh
    return (cost / n_tokens) * 1_000_000


# Snapshot dataclass

@dataclass
class EnergySnapshot:
    """Result of one measurement window."""
    joules: float
    elapsed_s: float
    avg_power_w: float
    sm_clock_mhz: int
    throttle_reasons: list[str]
    # Populated if n_tokens was provided
    n_tokens: int = 0
    tokens_per_joule: float = 0.0
    joules_per_1m_tokens: float = 0.0

    def cost_per_1m_tokens(self, price_per_kwh: float) -> float:
        return cost_per_million_tokens(self.joules, self.n_tokens, price_per_kwh)

    def to_dict(self) -> dict:
        return {
            "joules": round(self.joules, 3),
            "elapsed_s": round(self.elapsed_s, 4),
            "avg_power_w": round(self.avg_power_w, 1),
            "sm_clock_mhz": self.sm_clock_mhz,
            "throttle_reasons": self.throttle_reasons,
            "n_tokens": self.n_tokens,
            "tokens_per_joule": round(self.tokens_per_joule, 4),
            "joules_per_1m_tokens": round(self.joules_per_1m_tokens, 1),
        }


# Main tracker

class EnergyTracker:
    """
    GPU energy measurement via NVML's integrating energy counter.

    The counter (nvmlDeviceGetTotalEnergyConsumption) is a monotonic millijoule
    accumulator that runs at the hardware level.  Differencing two reads gives
    exact energy for the window — no sampling aliasing.
    """

    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._handle = None
        self._available = False
        self._start_mj: int = 0
        self._start_time: float = 0.0
        self._init_nvml()

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            # Verify the energy counter is readable
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            self._available = True
        except Exception as e:
            print(f"⚠ EnergyTracker: NVML energy counter unavailable ({e}). "
                  "Energy data will be zeros.")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def _read_energy_mj(self) -> int:
        if not self._available:
            return 0
        import pynvml
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)

    def _read_sm_clock(self) -> int:
        if not self._available:
            return -1
        try:
            import pynvml
            return pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
        except Exception:
            return -1

    def _read_throttle(self) -> int:
        if not self._available:
            return 0
        try:
            import pynvml
            return pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._handle)
        except Exception:
            return 0

    def _read_power_mw(self) -> int:
        if not self._available:
            return 0
        try:
            import pynvml
            return pynvml.nvmlDeviceGetPowerUsage(self._handle)
        except Exception:
            return 0

    # Manual start/stop API

    def start(self):
        """Begin an energy measurement window."""
        self._start_mj = self._read_energy_mj()
        self._start_time = time.monotonic()

    def stop(self, n_tokens: int = 0) -> EnergySnapshot:
        """End the window and return an EnergySnapshot."""
        end_mj = self._read_energy_mj()
        elapsed = time.monotonic() - self._start_time

        joules = (end_mj - self._start_mj) / 1000.0
        avg_power = joules / elapsed if elapsed > 0 else 0.0

        tok_per_j = n_tokens / joules if joules > 0 else 0.0
        j_per_1m = (joules / n_tokens * 1_000_000) if n_tokens > 0 else 0.0

        return EnergySnapshot(
            joules=joules,
            elapsed_s=elapsed,
            avg_power_w=avg_power,
            sm_clock_mhz=self._read_sm_clock(),
            throttle_reasons=decode_throttle_reasons(self._read_throttle()),
            n_tokens=n_tokens,
            tokens_per_joule=tok_per_j,
            joules_per_1m_tokens=j_per_1m,
        )

    # Context-manager API

    @contextmanager
    def region(self, n_tokens: int = 0):
        """
        Usage:
            with tracker.region(n_tokens=...) as snap_holder:
                # ... work ...
                pass
            snap = snap_holder.result
        """
        holder = _SnapHolder()
        self.start()
        try:
            yield holder
        finally:
            holder.result = self.stop(n_tokens=n_tokens)


class _SnapHolder:
    """Mutable container so the context manager can set .result after exit."""
    result: Optional[EnergySnapshot] = None


# Standalone CLI — quick power check

def main():
    import argparse

    p = argparse.ArgumentParser(description="Quick GPU energy/power check")
    p.add_argument("--seconds", type=float, default=3.0,
                   help="Measurement window (seconds)")
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    tracker = EnergyTracker(args.device)
    if not tracker.available:
        print("NVML energy counter not available. Exiting.")
        return

    print(f"Measuring for {args.seconds:.1f}s …")
    tracker.start()
    time.sleep(args.seconds)
    snap = tracker.stop()

    print(f"\nEnergy     : {snap.joules:.2f} J")
    print(f"Elapsed    : {snap.elapsed_s:.2f} s")
    print(f"Avg power  : {snap.avg_power_w:.1f} W")
    print(f"SM clock   : {snap.sm_clock_mhz} MHz")
    print(f"Throttle   : {', '.join(snap.throttle_reasons)}")

    # Also show instantaneous power for reference
    import pynvml
    instant_w = pynvml.nvmlDeviceGetPowerUsage(tracker._handle) / 1000.0
    print(f"Instant pwr: {instant_w:.1f} W")


if __name__ == "__main__":
    main()
