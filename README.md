#  Doctor Assistant Chatbot

This project now keeps only the main Streamlit application and the files it needs to run.

## 🎬 Preview Demo

<video src="Preview%20Demo.mp4" autoplay loop muted playsinline controls>
  Your browser does not support the video tag.
</video>

*(The video above will autoplay as an animation. Ensure `Preview Demo.mp4` is in the same folder as this README!)*

## 🚀 Local Setup Guide

Follow these steps to run the application on your local machine using a virtual environment.

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd kunal-medical-bot
```

### 2. Create and activate a virtual environment

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

3. Add your Groq API key in `key.env`:

```env
GROQ_API_KEY=your_groq_api_key
```

Security note: never commit real keys. A template is provided in `key.env.example`.

4. Start the app:

```bash
streamlit run main.py
```

5. Open the browser at:

```text
http://127.0.0.1:8501
```

## Required Runtime Files

- `medical_chatbot_ui.py`
- `custom_favicon.py`
- `assets/`
- `vectorstore/`
- `requirements.txt`
- `key.env`

## Notes

- The vector database inside `vectorstore/db_faiss/` must remain present.
- If `key.env` is missing or the API key is invalid, the app will not answer queries.
