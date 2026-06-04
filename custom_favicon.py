from pathlib import Path
from urllib.parse import quote

import streamlit as st


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
FAVICON_PATH = ASSETS_DIR / "medibot_favicon.svg"
FAVICON_DATA_URI = f"data:image/svg+xml,{quote(FAVICON_PATH.read_text(encoding='utf-8'))}"


def apply_custom_favicon() -> None:
    st.markdown(
        f"""
        <script>
        (function() {{
            const href = "{FAVICON_DATA_URI}";
            let favicon = document.querySelector("link[rel='icon']");
            if (!favicon) {{
                favicon = document.createElement("link");
                favicon.rel = "icon";
                document.head.appendChild(favicon);
            }}
            favicon.type = "image/svg+xml";
            favicon.href = href;
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )
