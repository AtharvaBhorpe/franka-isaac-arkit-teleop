"""teleop_arkit.core.cameras — camera-source spec parsing (shared by recorder + inference)."""


def parse_cameras(spec: str) -> list[tuple[str, str, str]]:
    """'wrist=/wrist_cam/image_raw scene=usb:0' -> [(name, kind, source)].

    kind = 'ros' if source looks like a topic (starts with '/'); else 'usb' (source
    'usb:N' or a device path) opened via cv2.VideoCapture.
    """
    cams = []
    for entry in spec.split():
        name, _, source = entry.partition("=")
        if not name or not source:
            raise ValueError(f"bad --cameras entry '{entry}', expected name=source")
        if source.startswith("/"):
            cams.append((name, "ros", source))
        else:
            cams.append((name, "usb", source[4:] if source.startswith("usb:") else source))
    return cams


def preprocess_image(bgr, target_hw):
    """Resize a BGR image, convert to RGB, and return as a normalized float CHW Tensor.
    
    cv2.resize expects (width, height), which is (target_hw[1], target_hw[0]).
    """
    import cv2
    import numpy as np
    import torch

    r = cv2.resize(bgr, (target_hw[1], target_hw[0]))
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0

