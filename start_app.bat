@echo off
setlocal
title Indian Legal QA (RAG) - launcher
cd /d "%~dp0"

rem ---- 0. Use the model folder on D: (setx does not reach programs that were already running) ----
if exist "D:\ollama\models" set OLLAMA_MODELS=D:\ollama\models

echo ==================================================
echo   Indian Legal QA (RAG) - starting up
echo ==================================================
echo.

if not exist "venv\Scripts\activate.bat" goto no_venv

rem ---- 1. Restart Ollama so it definitely uses the model folder on D: ----
where ollama >nul 2>&1
if errorlevel 1 goto no_ollama

echo Restarting Ollama with the model folder on D: ...
taskkill /IM "ollama app.exe" /F >nul 2>&1
taskkill /IM ollama.exe /F >nul 2>&1
timeout /t 2 /nobreak >nul
start "Ollama server" /min ollama serve
set tries=0

:wait_ollama
curl -s http://localhost:11434 >nul 2>&1
if %errorlevel%==0 goto ollama_ready
set /a tries+=1
if %tries% GEQ 40 goto ollama_timeout
timeout /t 1 /nobreak >nul
goto wait_ollama

:ollama_ready
echo Ollama is running.

rem ---- 2. Check that the RUNNING Ollama can see the model ----
curl -s http://localhost:11434/api/tags | findstr /i "llama3.2:3b" >nul
if errorlevel 1 goto model_missing

:start_app
rem ---- 3. Stop Streamlit asking for an email on first run ----
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  mkdir "%USERPROFILE%\.streamlit" 2>nul
  > "%USERPROFILE%\.streamlit\credentials.toml" (
    echo [general]
    echo email = ""
  )
)

rem ---- 4. Start the app ----
call venv\Scripts\activate.bat
set LLM_BACKEND=ollama
echo.
echo Opening the app in your browser: http://localhost:8501
echo The first start can take a minute while the models load.
echo KEEP THIS WINDOW OPEN while you use the app. Close it to stop the app.
echo.
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
echo.
echo The app has stopped.
pause
exit /b 0

:model_missing
echo.
echo [PROBLEM] Ollama is running, but the model llama3.2:3b is not available to it.
echo This usually means an older Ollama process is using a different model folder.
echo FIX: close this window, double-click stop_ollama.bat, then run this launcher again.
echo Or download the model now (about 2 GB).
echo.
choice /c YN /m "Download the model now"
if errorlevel 2 goto give_up
ollama pull llama3.2:3b
goto start_app

:give_up
echo.
pause
exit /b 1

:no_venv
echo [ERROR] Virtual environment not found in: %cd%\venv
echo Create it and install the requirements first (see README.md).
pause
exit /b 1

:no_ollama
echo [ERROR] Ollama was not found. Install it from https://ollama.com
echo then run this launcher again.
pause
exit /b 1

:ollama_timeout
echo [ERROR] Ollama did not start within 40 seconds.
echo Open the Ollama app from the Start menu, wait 10 seconds, and try again.
pause
exit /b 1
