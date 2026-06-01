import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
from core.models.image_detector import ImageDetector


def show():

    st.markdown(z
        '<p class="sub-header">🖼️ Image Deepfake Detection</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        "Upload an image to detect if it's AI-generated or real content."
    )

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff']
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            st.markdown("### 📄 Image Info")

            st.write(f"Format: {image.format}")
            st.write(f"Size: {image.size}")

        with col2:

            st.markdown("### ⚙️ Detection")

            if st.button("🔍 Run Detection"):

                with st.spinner("Analyzing image..."):

                    detector = ImageDetector()

                    result = detector.predict(image)

                    st.session_state.current_result = result

                    st.session_state.detection_history.append(result)

        if st.session_state.current_result:

            result = st.session_state.current_result

            st.markdown("---")

            st.markdown("## 📊 Detection Results")

            if result["is_ai_generated"]:

                st.error(
                    f"⚠️ AI GENERATED "
                    f"({result['confidence']*100:.2f}%)"
                )

            else:

                st.success(
                    f"✅ REAL IMAGE "
                    f"({result['confidence']*100:.2f}%)"
                )

            st.markdown("### AI Probability")

            st.progress(float(result["ai_probability"]))

            st.write(
                f"{result['ai_probability']*100:.2f}%"
            )

            st.markdown("### Real Probability")

            st.progress(float(result["real_probability"]))

            st.write(
                f"{result['real_probability']*100:.2f}%"
            )