import subprocess
import time
import webbrowser
import sys

python = sys.executable

# start FastAPI backend
subprocess.Popen([
    python, "-m", "uvicorn", "app:app"
])

# wait for backend to start
time.sleep(4)

# start Streamlit UI
subprocess.Popen([
    python, "-m", "streamlit", "run", "streamlit_app.py"
])

# wait for UI
time.sleep(5)

# open app
webbrowser.open("http://localhost:8501")