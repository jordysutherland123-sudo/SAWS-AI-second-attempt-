import cv2
from pathlib import Path


def collect_radar_images(archive_root: Path):
    """Find radar image files in the archive and sort them for timelapse."""
    archive_root = Path(archive_root)
    files = sorted(archive_root.rglob("*.png"))
    return [p for p in files if p.is_file()]


def build_timelapse(archive_root: Path, output_path: Path, fps: int = 3) -> None:
    """Build a simple MP4 timelapse from stored radar images."""
    image_paths = collect_radar_images(archive_root)
    if not image_paths:
        raise ValueError("No radar images found in archive. Download radar images first.")

    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise ValueError(f"Unable to read image {image_paths[0]}")

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
