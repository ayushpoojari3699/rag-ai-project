@echo off
cd /d "%~dp0"

call rag-env\Scripts\activate

start cmd /k uvicorn backend.app:app --port 8000
timeout /t 4 >nul
start cmd /k streamlit run ui\streamlit_app.py

timeout /t 6 >nul

"C:\Program Files\Google\Chrome\Application\chrome.exe" --app=http://localhost:8501 --user-data-dir="%LOCALAPPDATA%\DocuMindAI"