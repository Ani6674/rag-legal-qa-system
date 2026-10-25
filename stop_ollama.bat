@echo off
echo Stopping Ollama...
taskkill /IM "ollama app.exe" /F >nul 2>&1
taskkill /IM ollama.exe /F >nul 2>&1
echo Done. Ollama is stopped.
pause
