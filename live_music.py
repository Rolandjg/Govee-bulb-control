#!/usr/bin/env python3
"""
Pulse your Govee H6001 light brightness/color from the live audio currently
playing on your computer.

This version does NOT take an audio file. It records your system's monitor
source using parec, which works with PulseAudio and usually PipeWire-Pulse.

Requirements:
    pip install numpy bleak

Also needs:
    pactl
    parec

On Fedora/PipeWire these usually come from:
    sudo dnf install pipewire-pulseaudio pulseaudio-utils

Usage:
    python3 govee_live_pulse.py --device AA:BB:CC:DD:EE:FF

To find your bulb's MAC address:
    bluetoothctl scan on   (look for "ihoment_H6001_XXXX")
"""

import argparse
import asyncio
import math
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
import json

import numpy as np

try:
    from bleak import BleakClient, BleakError
except ImportError:
    sys.exit("Missing dependency. Install with:\n  pip install bleak")


# ─── BLE ──────────────────────────────────────────────────────────────────────

GOVEE_CHAR_UUID = "00010203-0405-0607-0809-0a0b0c0d2b11"


def xor_checksum(payload: list[int]) -> int:
    result = 0
    for b in payload:
        result ^= b
    return result


def make_packet(cmd: list[int]) -> bytearray:
    payload = cmd + [0x00] * (19 - len(cmd))
    payload.append(xor_checksum(payload))
    return bytearray(payload)


def power_packet(on: bool) -> bytearray:
    return make_packet([0x33, 0x01, 0x01 if on else 0x00])


def brightness_packet(brightness: int) -> bytearray:
    brightness = max(0, min(100, int(brightness)))

    if brightness == 0:
        return power_packet(False)

    bri_255 = max(1, min(255, int(brightness / 100 * 255)))
    return make_packet([0x33, 0x04, bri_255])


def color_packet(r: int, g: int, b: int) -> bytearray:
    return make_packet([
        0x33, 0x05, 0x0d,
        max(0, min(255, int(r))),
        max(0, min(255, int(g))),
        max(0, min(255, int(b))),
    ])


# ─── COLOR / THRESHOLDS ───────────────────────────────────────────────────────

def chroma_to_rgb(
    pitch: float,
    volume: float,
    cfg: argparse.Namespace,
    freq_threshold: float,
    vol_threshold: float,
) -> tuple[int, int, int]:
    """Select color from 4 quadrants: (low|high freq) x (low|high volume)."""
    high_freq = pitch >= freq_threshold
    high_vol = volume >= vol_threshold

    if high_freq and high_vol:
        return cfg.color_hf_hv
    elif high_freq and not high_vol:
        return cfg.color_hf_lv
    elif not high_freq and high_vol:
        return cfg.color_lf_hv
    else:
        return cfg.color_lf_lv


def count_color_quadrants(
    pitch_curve: np.ndarray,
    brightness_curve: np.ndarray,
    freq_threshold: float,
    vol_threshold: float,
) -> np.ndarray:
    """Return counts in order: LF/LV, LF/HV, HF/LV, HF/HV."""
    high_freq = pitch_curve >= freq_threshold
    high_vol = brightness_curve >= vol_threshold

    return np.array([
        np.sum(~high_freq & ~high_vol),  # LF + LV
        np.sum(~high_freq &  high_vol),  # LF + HV
        np.sum( high_freq & ~high_vol),  # HF + LV
        np.sum( high_freq &  high_vol),  # HF + HV
    ])


def find_balanced_thresholds(
    pitch_curve: np.ndarray,
    brightness_curve: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """
    Rolling/live approximation of the offline threshold search.
    Uses recent live frames instead of the whole song.
    """
    target = len(pitch_curve) / 4
    best_score = float("inf")
    best_freq = 0.5
    best_vol = 0.5
    best_counts = np.zeros(4, dtype=int)

    # Fewer steps than the offline version so this stays cheap while live.
    for fq in np.linspace(0.15, 0.85, 21):
        freq_t = float(np.quantile(pitch_curve, fq))

        for vq in np.linspace(0.15, 0.85, 21):
            vol_t = float(np.quantile(brightness_curve, vq))

            counts = count_color_quadrants(
                pitch_curve,
                brightness_curve,
                freq_t,
                vol_t,
            )

            score = np.abs(counts - target).sum()

            min_count = counts.min()
            if min_count < target * 0.08:
                score += (target * 0.08 - min_count) * 3

            if score < best_score:
                best_score = score
                best_freq = freq_t
                best_vol = vol_t
                best_counts = counts

    return best_freq, best_vol, best_counts


class RollingThresholds:
    """Adaptive thresholds from recent live frames."""

    def __init__(self, cfg: argparse.Namespace):
        maxlen = max(8, int(cfg.balance_window * cfg.update_hz))
        self.pitch_history = deque(maxlen=maxlen)
        self.volume_history = deque(maxlen=maxlen)
        self.freq_threshold = cfg.freq_threshold
        self.vol_threshold = cfg.vol_threshold
        self.last_recalc = 0.0

    def update(self, pitch: float, volume: float, cfg: argparse.Namespace) -> tuple[float, float]:
        self.pitch_history.append(float(pitch))
        self.volume_history.append(float(volume))

        if cfg.color_balance == "fixed":
            return cfg.freq_threshold, cfg.vol_threshold

        now = time.monotonic()
        if now - self.last_recalc < cfg.balance_recalc_interval:
            return self.freq_threshold, self.vol_threshold

        self.last_recalc = now

        if len(self.pitch_history) < max(8, cfg.update_hz):
            return self.freq_threshold, self.vol_threshold

        pitch_arr = np.asarray(self.pitch_history, dtype=np.float32)
        volume_arr = np.asarray(self.volume_history, dtype=np.float32)

        if cfg.color_balance == "median":
            self.freq_threshold = float(np.quantile(pitch_arr, 0.5))
            self.vol_threshold = float(np.quantile(volume_arr, 0.5))
        elif cfg.color_balance == "search":
            self.freq_threshold, self.vol_threshold, _ = find_balanced_thresholds(
                pitch_arr,
                volume_arr,
            )
        else:
            raise ValueError(f"Unknown color balance mode: {cfg.color_balance}")

        return self.freq_threshold, self.vol_threshold


# ─── LIVE AUDIO ───────────────────────────────────────────────────────────────

def get_default_monitor_source() -> str:
    if not shutil.which("pactl"):
        sys.exit("[error] pactl not found. Install pulseaudio-utils.")

    try:
        sink = subprocess.check_output(
            ["pactl", "get-default-sink"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as e:
        sys.exit(f"[error] Could not get default sink with pactl: {e}")

    return sink + ".monitor"


def sigmoid01(x: float, k: float) -> float:
    """Sigmoid shaped value, remapped so x=0 -> 0 and x=1 -> 1."""
    x = max(0.0, min(1.0, float(x)))

    if k <= 0:
        return x

    lo = 1.0 / (1.0 + math.exp(-k * (0.0 - 0.5)))
    hi = 1.0 / (1.0 + math.exp(-k * (1.0 - 0.5)))
    y = 1.0 / (1.0 + math.exp(-k * (x - 0.5)))

    return max(0.0, min(1.0, (y - lo) / (hi - lo + 1e-9)))


class LiveAudioAnalyzer:
    """Reads live speaker output and extracts volume, pitch, treble, and beats."""

    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.source = cfg.source or get_default_monitor_source()
        self.proc: subprocess.Popen | None = None
        self.running = False
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.state = {
            "t": time.monotonic(),
            "volume_raw": 0.0,
            "brightness_norm": 0.0,
            "pitch": 0.0,
            "treble": 0.0,
            "phase": -1,
            "rms": 0.0,
        }

    def start(self):
        if not shutil.which("parec"):
            sys.exit("[error] parec not found. Install pulseaudio-utils.")

        cmd = [
            "parec",
            "-d", self.source,
            "--format=float32le",
            "--channels", str(self.cfg.channels),
            f"--rate={self.cfg.audio_rate}",
            f"--latency-msec={self.cfg.latency_msec}",
        ]

        print(f"[audio] Recording live speaker output from: {self.source}")

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

        self.running = True
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()

    def read_state(self) -> dict:
        with self.lock:
            return dict(self.state)

    def _audio_loop(self):
        cfg = self.cfg
        chunk_frames = max(128, int(cfg.audio_rate / cfg.update_hz))
        byte_count = chunk_frames * cfg.channels * 4  # float32

        rms_peak = max(cfg.noise_gate * 20.0, 1e-4)
        volume_env = 0.0
        treble_env = 0.0
        pitch_smooth = 0.0
        energy_avg = max(cfg.noise_gate * 4.0, 1e-5)

        beat_count = 0
        last_beat_t = 0.0

        while self.running and self.proc and self.proc.stdout:
            raw = self.proc.stdout.read(byte_count)
            if not raw:
                continue

            samples = np.frombuffer(raw, dtype=np.float32)
            usable = (samples.size // cfg.channels) * cfg.channels
            if usable <= 0:
                continue

            samples = samples[:usable].reshape(-1, cfg.channels)
            mono = samples.mean(axis=1).astype(np.float32)

            if mono.size < 8:
                continue

            # Remove DC offset before RMS/FFT.
            mono = mono - float(np.mean(mono))
            rms = float(np.sqrt(np.mean(mono * mono)))
            now = time.monotonic()

            # ── Volume normalization ────────────────────────────────────────
            if rms < cfg.noise_gate:
                volume_raw = 0.0
            else:
                if rms > rms_peak:
                    rms_peak = rms
                else:
                    rms_peak = max(rms_peak * cfg.peak_decay, cfg.noise_gate * 2.0)

                volume_raw = min(1.0, rms / max(rms_peak, cfg.noise_gate * 2.0))
                volume_raw = volume_raw ** cfg.volume_power

            # Same style as the offline norm_decay(): value attacks immediately,
            # then falls according to --volume-decay.
            volume_env = max(volume_raw, volume_env * cfg.volume_decay)
            brightness_norm = sigmoid01(volume_env, cfg.sigmoid_steepness)

            # ── Spectrum / pitch / treble ───────────────────────────────────
            window = np.hanning(mono.size).astype(np.float32)
            spectrum = np.abs(np.fft.rfft(mono * window))
            freqs = np.fft.rfftfreq(mono.size, d=1.0 / cfg.audio_rate)
            mag_sum = float(np.sum(spectrum))

            if mag_sum <= 1e-9 or rms < cfg.noise_gate:
                centroid_norm = 0.0
                treble_raw = 0.0
            else:
                centroid = float(np.sum(freqs * spectrum) / mag_sum)
                centroid_norm = max(0.0, min(1.0, centroid / cfg.centroid_max_freq))

                treble_mask = freqs >= cfg.treble_freq
                treble_raw = float(np.sum(spectrum[treble_mask]) / (mag_sum + 1e-9))
                treble_raw = max(0.0, min(1.0, treble_raw))

            dt = mono.size / cfg.audio_rate
            alpha = 1.0 - math.exp(-dt / max(cfg.pitch_smoothing, 1e-6))
            pitch_smooth = pitch_smooth * (1.0 - alpha) + centroid_norm * alpha

            treble_env = max(treble_raw, treble_env * cfg.transient_decay)

            # ── Very simple live beat/onset estimate ─────────────────────────
            energy_avg = energy_avg * 0.95 + rms * 0.05
            phase = -1

            if cfg.beat_phase:
                loud_enough = volume_raw >= cfg.beat_threshold
                above_average = rms >= energy_avg * cfg.beat_sensitivity
                cooldown_done = (now - last_beat_t) >= cfg.beat_cooldown

                if loud_enough and above_average and cooldown_done:
                    phase = beat_count % 4
                    beat_count += 1
                    last_beat_t = now

            with self.lock:
                self.state = {
                    "t": now,
                    "volume_raw": volume_raw,
                    "brightness_norm": brightness_norm,
                    "pitch": pitch_smooth,
                    "treble": treble_env,
                    "phase": phase,
                    "rms": rms,
                }

    def stop(self):
        self.running = False

        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ─── LIGHT SHOW ───────────────────────────────────────────────────────────────

async def run_show(cfg: argparse.Namespace):
    frame_duration = 1.0 / cfg.update_hz

    audio = LiveAudioAnalyzer(cfg)
    audio.start()

    delay_buffer_len = max(4, int((cfg.sync_offset + 2.0) * cfg.update_hz))
    delay_buffer = deque(maxlen=delay_buffer_len)
    thresholds = RollingThresholds(cfg)

    print(f"\n[ble] Connecting to {cfg.device} ...")

    async with BleakClient(
        cfg.device,
        timeout=cfg.ble_timeout,
        pair_before_connect=False,
    ) as client:
        if not client.is_connected:
            print("[error] Failed to connect.")
            audio.stop()
            return

        print("[ok] Connected! Press Ctrl+C to stop.\n")

        await client.write_gatt_char(GOVEE_CHAR_UUID, power_packet(True), response=True)
        await asyncio.sleep(0.3)

        await client.write_gatt_char(
            GOVEE_CHAR_UUID,
            brightness_packet(cfg.max_brightness),
            response=True,
        )
        await asyncio.sleep(0.2)

        start_wall = time.monotonic()
        last_bri = -1
        last_rgb = None
        last_send_time = 0.0

        try:
            while True:
                tick_start = time.monotonic()

                live_state = audio.read_state()
                delay_buffer.append(live_state)

                wanted_t = tick_start - cfg.sync_offset
                state = delay_buffer[0] if delay_buffer else live_state

                # Choose the newest frame at or before wanted_t.
                for candidate in reversed(delay_buffer):
                    if candidate["t"] <= wanted_t:
                        state = candidate
                        break

                norm = float(state["brightness_norm"])
                pitch = float(state["pitch"])
                treble = float(state["treble"])
                phase = int(state["phase"])

                freq_t, vol_t = thresholds.update(pitch, norm, cfg)
                current_rgb = chroma_to_rgb(pitch, norm, cfg, freq_t, vol_t)

                flash_bri = None

                if cfg.beat_phase and phase >= 0:
                    if phase == 2 and cfg.beat3_flash:
                        flash_bri = cfg.max_brightness

                if treble > cfg.treble_strobe_threshold:
                    flash_bri = cfg.max_brightness

                if flash_bri is not None:
                    bri = cfg.max_brightness
                else:
                    bri = int(
                        cfg.min_brightness
                        + norm * (cfg.max_brightness - cfg.min_brightness)
                    )

                force_send = (tick_start - last_send_time) >= cfg.keepalive_interval
                changed = (bri != last_bri) or (current_rgb != last_rgb) or (flash_bri is not None)

                try:
                    if changed:
                        await client.write_gatt_char(
                            GOVEE_CHAR_UUID,
                            color_packet(*current_rgb),
                            response=False,
                        )
                        await client.write_gatt_char(
                            GOVEE_CHAR_UUID,
                            brightness_packet(bri),
                            response=False,
                        )

                        last_bri = bri
                        last_rgb = current_rgb
                        last_send_time = tick_start

                    elif force_send:
                        # Heartbeat/keep-alive: the bulb can drop the BLE link if it
                        # receives nothing for several seconds.
                        await client.write_gatt_char(
                            GOVEE_CHAR_UUID,
                            brightness_packet(last_bri if last_bri >= 0 else bri),
                            response=False,
                        )
                        last_send_time = tick_start

                except BleakError as e:
                    print(f"\n[warn] BLE write failed: {e}")

                elapsed = time.monotonic() - start_wall
                bar = "█" * int(norm * 24)
                phase_label = ["B1", "B2", "B3", "B4"][phase] if phase >= 0 else "  "

                print(
                    f"\r  {elapsed:6.1f}s  {bar:<24}  {bri:>3}%  {phase_label}  "
                    f"rgb{current_rgb}  pitch={pitch:.2f} vol={norm:.2f} treble={treble:.2f}",
                    end="",
                    flush=True,
                )

                elapsed_frame = time.monotonic() - tick_start
                await asyncio.sleep(max(0.0, frame_duration - elapsed_frame))

        except KeyboardInterrupt:
            print("\n\nStopped.")

        finally:
            audio.stop()

            try:
                await client.write_gatt_char(
                    GOVEE_CHAR_UUID,
                    brightness_packet(100),
                    response=True,
                )
            except BleakError:
                pass


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def parse_color(value: str) -> tuple[int, int, int]:
    """
    Parse a color string — either hex #RRGGBB / RRGGBB or R,G,B —
    into an (R, G, B) tuple.
    """
    stripped = value.strip().lstrip("#")

    if len(stripped) == 6 and all(c in "0123456789abcdefABCDEF" for c in stripped):
        r = int(stripped[0:2], 16)
        g = int(stripped[2:4], 16)
        b = int(stripped[4:6], 16)
        return (r, g, b)

    try:
        parts = [int(x.strip()) for x in value.split(",")]

        if len(parts) != 3 or not all(0 <= p <= 255 for p in parts):
            raise ValueError

        return tuple(parts)  # type: ignore[return-value]

    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Color must be #RRGGBB, RRGGBB, or R,G,B with values 0–255. Got: {value!r}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pulse a Govee H6001 light to live computer audio output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--device",
        "-d",
        required=True,
        help="Bulb MAC address. Find with: bluetoothctl scan on",
    )

    p.add_argument(
        "--preset",
        help="Path to a JSON preset file. Command-line flags override preset values.",
    )

    # ── Live audio input ──────────────────────────────────────────────────────
    aud = p.add_argument_group("live audio")
    aud.add_argument(
        "--source",
        help=(
            "PulseAudio/PipeWire source to record from. "
            "Defaults to the current default sink monitor, e.g. alsa_output...monitor"
        ),
    )
    aud.add_argument(
        "--audio-rate",
        type=int,
        default=44100,
        metavar="HZ",
        help="Audio capture sample rate",
    )
    aud.add_argument(
        "--channels",
        type=int,
        default=2,
        metavar="N",
        help="Audio capture channel count",
    )
    aud.add_argument(
        "--latency-msec",
        type=int,
        default=20,
        metavar="MS",
        help="parec capture latency",
    )
    aud.add_argument(
        "--noise-gate",
        type=float,
        default=0.0005,
        metavar="RMS",
        help="Ignore audio below this RMS level",
    )
    aud.add_argument(
        "--peak-decay",
        type=float,
        default=0.995,
        metavar="0.0–1.0",
        help="Decay for live RMS peak tracking. Higher = slower adaptation",
    )
    aud.add_argument(
        "--volume-power",
        type=float,
        default=0.8,
        metavar="P",
        help="Perceptual curve for volume. Lower than 1 boosts quiet audio",
    )
    aud.add_argument(
        "--centroid-max-freq",
        type=float,
        default=6000.0,
        metavar="HZ",
        help="Frequency mapped to pitch=1.0 for color selection",
    )
    aud.add_argument(
        "--treble-freq",
        type=float,
        default=4000.0,
        metavar="HZ",
        help="Frequency where treble energy starts",
    )

    # ── Brightness ────────────────────────────────────────────────────────────
    bri = p.add_argument_group("brightness")
    bri.add_argument(
        "--min-brightness",
        type=int,
        default=0,
        metavar="0–100",
        help="Minimum brightness during quiet sections",
    )
    bri.add_argument(
        "--max-brightness",
        type=int,
        default=100,
        metavar="0–100",
        help="Maximum brightness at peaks",
    )
    bri.add_argument(
        "--sigmoid-steepness",
        type=float,
        default=10.0,
        metavar="K",
        help=(
            "Sigmoid curve steepness for brightness shaping. "
            "Higher = more aggressive snap to min/max"
        ),
    )

    # ── Color ─────────────────────────────────────────────────────────────────
    col = p.add_argument_group("color: 4 quadrants, freq × volume")
    col.add_argument(
        "--color-lf-lv",
        type=parse_color,
        default=(180, 0, 255),
        metavar="R,G,B",
        help="Low frequency + low volume, e.g. quiet bass rumble",
    )
    col.add_argument(
        "--color-lf-hv",
        type=parse_color,
        default=(255, 60, 0),
        metavar="R,G,B",
        help="Low frequency + high volume, e.g. loud kick/bass hit",
    )
    col.add_argument(
        "--color-hf-lv",
        type=parse_color,
        default=(0, 180, 255),
        metavar="R,G,B",
        help="High frequency + low volume, e.g. soft shimmer/hi-hat",
    )
    col.add_argument(
        "--color-hf-hv",
        type=parse_color,
        default=(255, 255, 255),
        metavar="R,G,B",
        help="High frequency + high volume, e.g. bright crash/snare",
    )
    col.add_argument(
        "--freq-threshold",
        type=float,
        default=0.5,
        metavar="0.0–1.0",
        help="Normalized spectral centroid threshold. Used directly with --color-balance fixed",
    )
    col.add_argument(
        "--vol-threshold",
        type=float,
        default=0.5,
        metavar="0.0–1.0",
        help="Normalized volume threshold. Used directly with --color-balance fixed",
    )
    col.add_argument(
        "--color-balance",
        choices=["search", "median", "fixed"],
        default="fixed",
        help=(
            "How to choose color thresholds from recent live audio. "
            "search = closest rolling 4-way split, median = rolling medians, "
            "fixed = use freq/vol thresholds directly"
        ),
    )
    col.add_argument(
        "--balance-window",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Rolling history length for live median/search color balancing",
    )
    col.add_argument(
        "--balance-recalc-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="How often to recalculate adaptive color thresholds",
    )
    col.add_argument(
        "--pitch-smoothing",
        type=float,
        default=0.1,
        metavar="SECONDS",
        help="Smoothing time constant for spectral centroid before color selection",
    )

    # ── Timing & sync ─────────────────────────────────────────────────────────
    tim = p.add_argument_group("timing")
    tim.add_argument(
        "--update-hz",
        type=int,
        default=12,
        metavar="HZ",
        help="BLE update rate. Practical ceiling is usually around 15",
    )
    tim.add_argument(
        "--sync-offset",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Delay lights behind live audio by this many seconds. "
            "Must be >= 0 because live audio cannot be predicted early"
        ),
    )
    tim.add_argument(
        "--keepalive-interval",
        type=float,
        default=4.5,
        metavar="SECONDS",
        help="Force a BLE write at least this often to avoid bulb disconnects",
    )

    # ── Beat effects ──────────────────────────────────────────────────────────
    beat = p.add_argument_group("beat effects")
    beat.add_argument(
        "--beat-phase",
        dest="beat_phase",
        action="store_true",
        default=False,
        help="Enable simple live B1–B4 beat phase tracking",
    )
    beat.add_argument(
        "--no-beat-phase",
        dest="beat_phase",
        action="store_false",
        help="Disable live beat phase tracking",
    )
    beat.add_argument(
        "--no-beat3-flash",
        dest="beat3_flash",
        action="store_false",
        default=True,
        help="Disable the brightness flash on beat 3 of each bar",
    )
    beat.add_argument(
        "--beat-threshold",
        type=float,
        default=0.62,
        metavar="0.0–1.0",
        help="Normalized volume required to count a live beat",
    )
    beat.add_argument(
        "--beat-sensitivity",
        type=float,
        default=1.35,
        metavar="RATIO",
        help="Beat must exceed rolling average energy by this ratio",
    )
    beat.add_argument(
        "--beat-cooldown",
        type=float,
        default=0.25,
        metavar="SECONDS",
        help="Minimum time between detected beats",
    )
    beat.add_argument(
        "--treble-strobe-threshold",
        type=float,
        default=1.0,
        metavar="0.0–1.0",
        help="Treble energy level required to trigger a strobe flash. 1.0 effectively disables it",
    )

    # ── Decay / envelope ─────────────────────────────────────────────────────
    env = p.add_argument_group("envelope")
    env.add_argument(
        "--bass-decay",
        type=float,
        default=0.28,
        metavar="0.0–1.0",
        help="Reserved for bass effects. Higher = longer thumpy tail",
    )
    env.add_argument(
        "--transient-decay",
        type=float,
        default=0.50,
        metavar="0.0–1.0",
        help="Transient/treble decay rate. Lower = snappier",
    )
    env.add_argument(
        "--sustain-decay",
        type=float,
        default=0.02,
        metavar="0.0–1.0",
        help="Reserved sustain/harmonic decay rate",
    )
    env.add_argument(
        "--volume-decay",
        type=float,
        default=0.15,
        metavar="0.0–1.0",
        help="Volume brightness decay. Higher = longer brightness tail",
    )

    # ── BLE connection ────────────────────────────────────────────────────────
    ble = p.add_argument_group("ble")
    ble.add_argument(
        "--ble-timeout",
        type=float,
        default=15.0,
        metavar="SECONDS",
        help="BLE connection timeout",
    )

    return p


def validate_cfg(cfg: argparse.Namespace):
    for attr, lo, hi in [
        ("min_brightness", 0, 100),
        ("max_brightness", 0, 100),
    ]:
        v = getattr(cfg, attr)
        if not (lo <= v <= hi):
            sys.exit(f"[error] --{attr.replace('_', '-')} must be {lo}–{hi}, got {v}")

    if cfg.min_brightness >= cfg.max_brightness:
        sys.exit("[error] --min-brightness must be less than --max-brightness")

    for attr in [
        "freq_threshold",
        "vol_threshold",
        "treble_strobe_threshold",
        "bass_decay",
        "transient_decay",
        "sustain_decay",
        "volume_decay",
        "peak_decay",
        "beat_threshold",
    ]:
        v = getattr(cfg, attr)
        if not (0.0 <= v <= 1.0):
            sys.exit(f"[error] --{attr.replace('_', '-')} must be 0.0–1.0, got {v}")

    if cfg.update_hz <= 0:
        sys.exit("[error] --update-hz must be greater than 0")

    if cfg.audio_rate <= 0:
        sys.exit("[error] --audio-rate must be greater than 0")

    if cfg.channels <= 0:
        sys.exit("[error] --channels must be greater than 0")

    if cfg.latency_msec <= 0:
        sys.exit("[error] --latency-msec must be greater than 0")

    if cfg.noise_gate < 0:
        sys.exit("[error] --noise-gate must be >= 0")

    if cfg.volume_power <= 0:
        sys.exit("[error] --volume-power must be greater than 0")

    if cfg.centroid_max_freq <= 0:
        sys.exit("[error] --centroid-max-freq must be greater than 0")

    if cfg.treble_freq <= 0:
        sys.exit("[error] --treble-freq must be greater than 0")

    if cfg.pitch_smoothing <= 0:
        sys.exit("[error] --pitch-smoothing must be greater than 0")

    if cfg.balance_window <= 0:
        sys.exit("[error] --balance-window must be greater than 0")

    if cfg.balance_recalc_interval <= 0:
        sys.exit("[error] --balance-recalc-interval must be greater than 0")

    if cfg.sync_offset < 0:
        sys.exit("[error] --sync-offset cannot be negative for live audio")

    if cfg.keepalive_interval <= 0:
        sys.exit("[error] --keepalive-interval must be greater than 0")

    if cfg.ble_timeout <= 0:
        sys.exit("[error] --ble-timeout must be greater than 0")

    if cfg.beat_sensitivity <= 0:
        sys.exit("[error] --beat-sensitivity must be greater than 0")

    if cfg.beat_cooldown < 0:
        sys.exit("[error] --beat-cooldown must be >= 0")

def load_preset_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    # First pass: only find --preset
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset")
    preset_cfg, remaining_args = preset_parser.parse_known_args()

    defaults = {}

    if preset_cfg.preset:
        with open(preset_cfg.preset, "r", encoding="utf-8") as f:
            defaults = json.load(f)

        # Convert JSON lists to tuples for color values
        for key in [
            "color_lf_lv",
            "color_lf_hv",
            "color_hf_lv",
            "color_hf_hv",
        ]:
            if key in defaults:
                defaults[key] = tuple(defaults[key])

        parser.set_defaults(**defaults)

    return parser.parse_args()

async def main_async(cfg: argparse.Namespace):
    validate_cfg(cfg)
    await run_show(cfg)


def main():
    parser = build_parser()
    cfg = load_preset_args(parser)
    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    main()
