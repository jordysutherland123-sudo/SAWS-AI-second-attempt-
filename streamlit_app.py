import pathlib
import sys
from datetime import date

import streamlit as st

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import (
    ALLOWED_EXTENSIONS,
    CLASS_LABELS,
    DAYS_TO_KEEP,
    LIVE_RADAR_PAGE_URL,
    RADAR_ARCHIVE_ROOT,
    RADAR_IMAGE_URL,
    RADAR_SITE_CODE,
    TIMELAPSE_OUTPUT,
)
from scripts import ai_classifier, download_radar, radar_timelapse


def get_archive_dates():
    dates = sorted(
        [p.name for p in RADAR_ARCHIVE_ROOT.iterdir() if p.is_dir()], reverse=True
    )
    return dates[:DAYS_TO_KEEP]


def show_image_preview(image_path):
    st.image(str(image_path), caption=image_path.name, use_column_width=True)


def main():
    st.set_page_config(page_title="Sydney Weather AI", layout="wide")
    st.title("Sydney Weather AI Dashboard")
    st.markdown(
        "This dashboard helps you view radar archive replay, track live storms, and test AI image classification for radar signatures."
    )

    with st.sidebar:
        st.header("Controls")
        st.markdown(f"Live radar site: [{LIVE_RADAR_PAGE_URL}]({LIVE_RADAR_PAGE_URL})")
        if st.button("Download live radar image"):
            download_radar.fetch_live_radar_from_page(
                LIVE_RADAR_PAGE_URL, RADAR_SITE_CODE, RADAR_ARCHIVE_ROOT
            )
            st.success("Downloaded latest radar image to archive.")

        if st.button("Build 7-day timelapse"):
            radar_timelapse.build_timelapse(RADAR_ARCHIVE_ROOT, TIMELAPSE_OUTPUT)
            st.success(f"Timelapse saved to {TIMELAPSE_OUTPUT}")

        st.markdown("---")
        st.markdown("### AI classification")
        st.write("Use a saved radar image or upload your own.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Radar archive replay")
        dates = get_archive_dates()
        if dates:
            selected_date = st.selectbox("Choose archive day", dates)
            day_folder = RADAR_ARCHIVE_ROOT / selected_date
            image_files = sorted(
                [p for p in day_folder.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS]
            )
            if image_files:
                show_image_preview(image_files[-1])
                if st.button("Show all images for selected day"):
                    for image_path in image_files[-8:]:
                        st.image(str(image_path), width=300)
            else:
                st.warning("No radar images found for this day yet.")
        else:
            st.info("No archive folders are present. Click the button in the sidebar to download live radar images.")

    with col2:
        st.subheader("Live radar and AI classification")
        st.markdown("Upload a radar snapshot or use the latest downloaded image.")
        uploaded_file = st.file_uploader(
            "Upload radar image", type=[ext.strip('.') for ext in ALLOWED_EXTENSIONS]
        )

        if uploaded_file:
            with st.spinner("Classifying uploaded image..."):
                predictions = ai_classifier.classify_uploaded_image(uploaded_file, CLASS_LABELS)
                st.write(predictions)
                st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

        elif dates:
            latest_day = dates[0]
            latest_folder = RADAR_ARCHIVE_ROOT / latest_day
            latest_images = sorted(
                [p for p in latest_folder.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS]
            )
            if latest_images:
                latest_path = latest_images[-1]
                st.write("Latest radar archive image:")
                show_image_preview(latest_path)
                if st.button("Run AI classifier on latest image"):
                    result = ai_classifier.classify_image_file(latest_path, CLASS_LABELS)
                    st.write(result)
            else:
                st.warning("No latest radar image available for classification.")
        else:
            st.warning("No radar archive images found yet.")

    st.markdown("---")
    st.write(
        "To train the AI model, add labeled radar images under `datasets/training_images/<label>/` and run `scripts/ai_classifier.py`."
    )


if __name__ == "__main__":
    main()
