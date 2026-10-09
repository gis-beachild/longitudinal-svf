"""Read the subject/session manifests used by the Babofet MRI datasets."""

import json
from pathlib import Path


def load_sessions(root_dir: str, manifest: str) -> list[list[dict]]:
    root = Path(root_dir)
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    with manifest_path.open() as stream:
        subjects = json.load(stream)["subjects"]

    result = []
    for subject in subjects:
        sessions = []
        for session in subject["sessions"]:
            # The manifests use a leading slash for paths relative to root_dir.
            image = root / session["image"].lstrip("/")
            label = root / session["segmentation"].lstrip("/")
            if not label.is_file():
                # Two Babofet manifests retain an older derivatives path for
                # sessions 08/09; their labels now live in parcellations/.
                current_label = root / "parcellations" / label.name
                if current_label.is_file():
                    label = current_label
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(f"Missing MRI or segmentation: {image}, {label}")
            sessions.append({"image": str(image), "label": str(label), "age": float(session["age"])})
        result.append(sorted(sessions, key=lambda item: item["age"]))
    return result
