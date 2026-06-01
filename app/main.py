"""DEEPVISION - Deepfake Detection System

Main Streamlit application entry point.
"""

import streamlit as st
from pathlib import Path
import sys


project_root = Path(__file__).parent.parent

sys.path.insert(0, str(project_root))

def show():
    """Display image detection page"""

    st.markdown(
        '<p class="sub-header">🖼️ Image Deepfake Detection</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        "Upload an image to detect if it's AI-generated or real content."
    )

    # Upload section
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'],
        help="Supported formats: JPG, JPEG, PNG, WebP, BMP, TIFF"
    )

    if uploaded_file is not None:

        # Display image
        col1, col2 = st.columns([1, 1])

        with col1:

            image = Image.open(uploaded_file)

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            # Image info
            st.markdown("### 📄 Image Info")

            info_col1, info_col2 = st.columns(2)

            with info_col1:
                st.metric("Format", image.format or "Unknown")
                st.metric(
                    "Size",
                    f"{image.size[0]} x {image.size[1]}"
                )

            with info_col2:
                st.metric("Mode", image.mode)

                file_size = len(uploaded_file.getvalue()) / 1024

                st.metric(
                    "File Size",
                    f"{file_size:.1f} KB"
                )

        with col2:

            # Detection controls
            st.markdown("### ⚙️ Detection Settings")

            use_ensemble = st.checkbox(
                "Use Ensemble Methods",
                value=True
            )

            analysis_depth = st.select_slider(
                "Analysis Depth",
                options=["Quick", "Standard", "Deep"],
                value="Standard"
            )

            show_details = st.checkbox(
                "Show Detailed Analysis",
                value=True
            )

            # Run detection
            if st.button("🔍 Run Detection", type="primary"):

                with st.spinner("Analyzing image..."):

                    # Load detector
                    detector = ImageDetector()

                    # Predict
                    result = detector.predict(image)

                    # Add metadata
                    result["file_name"] = uploaded_file.name
                    result["file_size"] = file_size
                    result["dimensions"] = image.size
                    result["timestamp"] = datetime.now().isoformat()

                    # Save session
                    st.session_state.current_result = result

                    st.session_state.detection_history.append(result)

        # Display results
        if st.session_state.current_result:

            result = st.session_state.current_result

            st.markdown("---")

            st.markdown("### 📊 Detection Results")

            # Main result card
            result_class = (
                "ai-result"
                if result["is_ai_generated"]
                else "real-result"
            )

            result_icon = (
                "⚠️"
                if result["is_ai_generated"]
                else "✅"
            )

            result_text = (
                "AI GENERATED"
                if result["is_ai_generated"]
                else "REAL CONTENT"
            )

            st.markdown(f"""
            <div class="result-card {result_class}"
                 style="text-align: center; padding: 2rem;">

                <h2 style="margin: 0; font-size: 2rem;">
                    {result_icon} {result_text}
                </h2>

                <p style="margin: 0.5rem 0;
                    color: var(--text-secondary);">

                    Confidence:
                    {result['confidence']*100:.1f}%

                </p>

            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("### 📈 Probability Breakdown")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("**AI Generated Probability**")

                st.progress(float(result["ai_probability"]))

                st.markdown(
                    f"**{result['ai_probability']*100:.1f}%**"
                )

            with col2:

                st.markdown("**Real Content Probability**")

                st.progress(float(result["real_probability"]))

                st.markdown(
                    f"**{result['real_probability']*100:.1f}%**"
                )

            # Detailed analysis
            if show_details:

                st.markdown("---")

                st.markdown("### 🔬 Detailed Analysis")

                tab1, tab2, tab3 = st.tabs([
                    "CNN Analysis",
                    "Image Statistics",
                    "Metadata"
                ])

                with tab1:

                    st.markdown(
                        "#### Neural Network Classification"
                    )

                    st.progress(float(result["confidence"]))

                    st.write(
                        "EfficientNet-B0 backbone"
                    )

                    st.metric(
                        "Confidence Score",
                        f"{result['confidence']*100:.1f}%"
                    )

                with tab2:

                    st.markdown(
                        "#### Image Statistics"
                    )

                    img_np = np.array(image)

                    st.metric(
                        "Mean Pixel Value",
                        f"{img_np.mean():.2f}"
                    )

                    st.metric(
                        "Std Deviation",
                        f"{img_np.std():.2f}"
                    )

                    st.metric(
                        "Resolution",
                        f"{image.size[0]}x{image.size[1]}"
                    )

                with tab3:

                    st.markdown(
                        "#### Metadata Verification"
                    )

                    exif = image.getexif()

                    st.metric(
                        "EXIF Tags",
                        len(exif)
                    )

                    st.metric(
                        "Color Mode",
                        image.mode
                    )

                    if len(exif) == 0:
                        st.warning(
                            "No EXIF metadata found"
                        )
                    else:
                        st.success(
                            "Metadata detected"
                        )

            # Export options
            st.markdown("---")

            st.markdown("### 💾 Export Results")

            export_col1, export_col2 = st.columns(2)

            with export_col1:

                if st.button("📄 Save JSON"):

                    import json

                    json_str = json.dumps(
                        result,
                        indent=2
                    )

                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name="detection_result.json",
                        mime="application/json"
                    )

            with export_col2:

                if st.button("🖨️ Generate Report"):

                    st.info(
                        "PDF report generation coming soon!"
                    )