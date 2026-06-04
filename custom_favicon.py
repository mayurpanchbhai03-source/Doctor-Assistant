from pathlib import Path
from urllib.parse import quote
import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
FAVICON_PATH = ASSETS_DIR / "medibot_favicon.svg"

# Safe fallback loading logic
if FAVICON_PATH.is_file():
    FAVICON_DATA_URI = f"data:image/svg+xml,{quote(FAVICON_PATH.read_text(encoding='utf-8'))}"
else:
    # Uses a standard medical cross emoji as a fallback icon asset
    FAVICON_DATA_URI = "data:image/svg+xml,%3Csvg xmlns='http://w3.org' viewBox='0 0 10 10'%3E%3Ctext y='9' font-size='9'%3E🏥%3C/text%3E%3C/svg%3E"

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
