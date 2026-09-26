"""Background threads for the phone (grid) camera.

Capture thread: grabs every frame and finds the board (fast), so the
video stays smooth. Scan thread: while `scanning`, reads the cards from
the newest frame whenever it's free (slow: SIFT), and feeds the votes.
The UI thread only calls snapshot(), so nothing slow ever blocks it.
"""

import threading
import time

import cv2

from .reader import BoardReader, SlotVotes


class GridCam:
    def __init__(self, index, reader: BoardReader, name="", scan_every_s=0.35):
        self.name = name
        self.cap = cv2.VideoCapture(index)       # opened on the main thread
        if not self.cap.isOpened():
            raise RuntimeError(f"grid camera [{index}] {name} didn't open")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.reader = reader
        if not reader.detect_every_s:
            reader.detect_every_s = 0.3       # board is static: ~3×/s is plenty
        self.votes = SlotVotes()
        self.scanning = True
        self.scan_every_s = scan_every_s
        self._lock = threading.Lock()
        self._frame = self._view = None
        self._fps = self.scan_s = 0.0
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=f, daemon=True)
                         for f in (self._capture, self._scan)]
        for th in self._threads:
            th.start()

    def _capture(self):
        last = time.monotonic()
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            now = time.monotonic()
            view = self.reader.locate(frame, now)
            with self._lock:
                self._frame, self._view = frame, view
                self._fps = 0.9 * self._fps + 0.1 / max(now - last, 1e-3)
            last = now

    def _scan(self):
        while not self._stop.is_set():
            with self._lock:
                frame, view, scanning = self._frame, self._view, self.scanning
            if not scanning or frame is None or not view.found or view.partial:
                time.sleep(0.1)
                continue
            t0 = time.monotonic()
            reading = self.reader.identify(frame, view)
            self.scan_s = time.monotonic() - t0
            with self._lock:
                if self.scanning:           # may have locked meanwhile
                    self.votes.add(reading)
            time.sleep(max(0.0, self.scan_every_s - self.scan_s))

    def snapshot(self):
        with self._lock:
            return self._frame, self._view, self._fps

    def rescan(self):
        with self._lock:
            self.votes = SlotVotes()
            self.scanning = True

    def lock(self):
        """Freeze the parties: stop reading cards (tracking continues)."""
        with self._lock:
            self.scanning = False
            return {s: self.votes.party(s) for s in ("player", "cpu")}

    def close(self):
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2)
        self.cap.release()
