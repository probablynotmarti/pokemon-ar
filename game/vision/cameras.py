"""Which OpenCV camera index is which device (macOS).

This OpenCV build (4.11, AVFoundation backend) takes devicesWithMediaType:
and sorts it by uniqueID, so that's the order VideoCapture(i) uses. The raw
AVFoundation order is different and reshuffles when OBS starts.
"""

import time


def list_cameras(wait_for_phone_s=0.0):
    """[(name, model)] in OpenCV index order. macOS discovers Continuity
    Camera iPhones in the background, so optionally keep looking for one
    for a few seconds before answering."""
    try:
        import AVFoundation as AV
    except ImportError:
        return []
    if wait_for_phone_s > 0:
        from Foundation import NSDate, NSRunLoop
        deadline = time.monotonic() + wait_for_phone_s
        while time.monotonic() < deadline:
            names = [f"{d.localizedName()} {d.modelID()}" for d in
                     AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)]
            if any(is_phone(n, "") for n in names):
                break
            NSRunLoop.currentRunLoop().runUntilDate_(
                NSDate.dateWithTimeIntervalSinceNow_(0.25))
    devs = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))
    devs += list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed))
    devs.sort(key=lambda d: str(d.uniqueID()))
    return [(str(d.localizedName()), str(d.modelID())) for d in devs]


def _text(name, model):
    return f"{name} {model}".lower()


def is_phone(name, model):
    return any(w in _text(name, model) for w in ("iphone", "ipad"))


def is_virtual(name, model):
    return any(w in _text(name, model) for w in ("obs", "virtual", "desk view"))


def pick_hand_camera(cams):
    """The Mac's own camera (the iPhone is saved for the grid)."""
    for want in ("facetime", "macbook", "built-in"):
        for i, (name, model) in enumerate(cams):
            if want in name.lower() and not (is_phone(name, model)
                                             or is_virtual(name, model)):
                return i
    for i, (name, model) in enumerate(cams):
        if not (is_phone(name, model) or is_virtual(name, model)):
            return i
    return 0


def pick_grid_camera(cams):
    """The iPhone via Continuity Camera, or None if it isn't connected."""
    for i, (name, model) in enumerate(cams):
        if is_phone(name, model):
            return i
    return None
