@echo off
cd /d "%~dp0"
echo Starting EGX dashboard... keep this window open while you use it.
streamlit run app.py
pause
