#!/usr/bin/env python3
"""
govee_desktop_ambient.py — Set Govee H6001 color from your X11 screen and
brightness from actual speaker output audio.

Requirements:
    pip install bleak numpy mss

Also needs:
    pactl
    parec

On Fedora/PipeWire these usually come from:
    sudo dnf install pipewire-pulseaudio pulseaudio-utils

Usage:
    python3 govee_desktop_ambient.py --device AA:BB:CC:DD:EE:FF
"""

import argparse
import asyncio
import random
import shutil
import subprocess
import sys
import time
from collections import deque

import threading
import math

import numpy as np
import mss

try:
    from bleak import BleakClient, BleakError
except ImportError:
    print("Missing dependency. Install with:\n  pip install bleak")
    sys.exit(1)


GOVEE_CHAR_UUID = "00010203-0405-0607-0809-0a0b0c0d2b11"

UPDATE_HZ = 12

MIN_BRIGHTNESS = 1
MAX_BRIGHTNESS = 100

SCREEN_SAMPLE_PIXELS = 300

AUDIO_RATE = 44100
AUDIO_CHUNK_FRAMES = 512
AUDIO_SMOOTHING = 0.0
AUDIO_GAIN = 12.0

COLOR_BLEND_SPEED = 0.35


def xor_checksum(payload: list) -> int:
    r = 0
    for b in payload:
        r ^= b
    return r


def make_packet(cmd: list) -> bytearray:
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


def blend_rgb(current, target, speed):
    return tuple(
        int(current[i] + (target[i] - current[i]) * speed)
        for i in range(3)
    )


def get_default_monitor_source():
    if not shutil.which("pactl"):
        print("[error] pactl not found. Install pulseaudio-utils.")
        sys.exit(1)

    sink = subprocess.check_output(
        ["pactl", "get-default-sink"],
        text=True
    ).strip()

    return sink + ".monitor"


class SpeakerMonitor:
    def __init__(self):
        self.source = get_default_monitor_source()
        self.proc = None
        self.level = 0.0
        self.running = False
        self.thread = None

    def start(self):
        if not shutil.which("parec"):
            print("[error] parec not found. Install pulseaudio-utils.")
            sys.exit(1)

        cmd = [
            "parec",
            "-d", self.source,
            "--format=float32le",
            "--channels=2",
            f"--rate={AUDIO_RATE}",
            "--latency-msec=20",
        ]

        print(f"[audio] Recording speaker output from: {self.source}")

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

        self.running = True
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()

    def _audio_loop(self):
        chunk_frames = 512
        byte_count = chunk_frames * 2 * 4

        smoothed = 0.0
        peak = 0.05  # baseline so silence isn't NaN

        while self.running and self.proc and self.proc.stdout:
            raw = self.proc.stdout.read(byte_count)
            if not raw:
                continue

            samples = np.frombuffer(raw, dtype=np.float32)
            if len(samples) == 0:
                continue

            rms = float(np.sqrt(np.mean(samples * samples)))

            # --- Peak tracking ---
            if rms > peak:
                peak = rms              # instant attack
            else:
                peak *= 0.995          # slow decay (tune this)

            # Normalize against recent peak
            level = rms / max(peak, 1e-5)
            level = min(1.0, level)

            # Optional perceptual curve
            level = level ** 0.8

            # Smooth (fast up, slower down)
            if level > smoothed:
                smoothed = smoothed * 0.3 + level * 0.7
            else:
                smoothed = smoothed * 0.9 + level * 0.1

            self.level = smoothed

    def read_level(self):
        return self.level

    def stop(self):
        self.running = False
        if self.proc:
            self.proc.terminate()


class ScreenSampler:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]

        width = self.monitor["width"]
        height = self.monitor["height"]

        self.sample_points = [
            (random.randrange(width), random.randrange(height))
            for _ in range(SCREEN_SAMPLE_PIXELS)
        ]

    def dominant_color(self):
        img = self.sct.grab(self.monitor)

        width = img.width
        pixels = []

        for x, y in self.sample_points:
            idx = (y * width + x) * 4

            b = img.raw[idx]
            g = img.raw[idx + 1]
            r = img.raw[idx + 2]

            pixels.append((r, g, b))

        arr = np.array(pixels, dtype=np.float32)

        r, g, b = np.median(arr, axis=0)

        return int(r), int(g), int(b)


async def run(device: str, use_audio: bool):
    frame_duration = 1.0 / UPDATE_HZ

    screen = ScreenSampler()
    audio = None
    if use_audio:
        audio = SpeakerMonitor()
        audio.start()

    print(f"[ble] Connecting to {device} ...")

    async with BleakClient(device, timeout=300) as client:
        if not client.is_connected:
            print("[error] Failed to connect.")
            return

        print("[ok] Connected. Press Ctrl+C to stop.")

        await client.write_gatt_char(GOVEE_CHAR_UUID, power_packet(True), response=True)
        await asyncio.sleep(0.2)

        current_rgb = (255, 255, 255)
        last_brightness = -1
        last_rgb = None

        last_send_time = 0

        try:
            while True:
                start = time.monotonic()
                now = start

                target_rgb = screen.dominant_color()
                current_rgb = blend_rgb(current_rgb, target_rgb, COLOR_BLEND_SPEED)


                if use_audio:
                    audio_level = audio.read_level()
                    brightness = int(
                        MIN_BRIGHTNESS
                        + audio_level * (MAX_BRIGHTNESS - MIN_BRIGHTNESS)
                    )
                else:
                    audio_level = 0.0
                    brightness = MAX_BRIGHTNESS  # or pick a fixed value like 50

                try:
                    sent = False

                    if current_rgb != last_rgb:
                        await client.write_gatt_char(
                            GOVEE_CHAR_UUID,
                            color_packet(*current_rgb),
                            response=False,
                        )
                        last_rgb = current_rgb
                        sent = True

                    if brightness != last_brightness:
                        await client.write_gatt_char(
                            GOVEE_CHAR_UUID,
                            brightness_packet(brightness),
                            response=False,
                        )
                        last_brightness = brightness
                        sent = True

                    # --- Heartbeat: ensure at least one write every ~5 seconds ---
                    if not sent and (now - last_send_time) > 4.5:
                        await client.write_gatt_char(
                            GOVEE_CHAR_UUID,
                            brightness_packet(last_brightness),
                            response=False,
                        )
                        sent = True

                    if sent:
                        last_send_time = now

                except BleakError as e:
                    print(f"\n[warn] BLE write failed: {e}")

                bar = "█" * int(audio_level * 24) if use_audio else ""
                print(
                    f"\rbrightness {brightness:>3}%  {bar:<24}  rgb{current_rgb}",
                    end="",
                    flush=True,
                )

                elapsed = time.monotonic() - start
                await asyncio.sleep(max(0.0, frame_duration - elapsed))

        except KeyboardInterrupt:
            print("\nStopped.")

        finally:
            if audio:
                audio.stop()
            await client.write_gatt_char(
                GOVEE_CHAR_UUID,
                brightness_packet(100),
                response=True,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Control a Govee H6001 from screen color and speaker output."
    )
    parser.add_argument(
        "--device",
        "-d",
        required=True,
        help="Bulb MAC address, find with: bluetoothctl scan on",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Enable audio-reactive brightness",
    )
    args = parser.parse_args()

    asyncio.run(run(args.device, args.audio))


if __name__ == "__main__":
    main()
