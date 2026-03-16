import subprocess
import time
import webbrowser
import sys

print("Starting FastAPI backend...")

backend = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

time.sleep(4)

print("Starting Streamlit frontend...")

frontend = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

time.sleep(4)

print("Opening browser...")

webbrowser.open("http://localhost:8501")

backend.wait()
frontend.wait()