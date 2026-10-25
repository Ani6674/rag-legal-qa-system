@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop'); $s=(New-Object -ComObject WScript.Shell).CreateShortcut($d+'\Indian Legal QA (RAG).lnk'); $s.TargetPath='%~dp0start_app.bat'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%SystemRoot%\System32\shell32.dll,13'; $s.Save(); Write-Host ('Shortcut created on: ' + $d)"
echo.
echo Done. Look for "Indian Legal QA (RAG)" on your Desktop and double-click it.
pause
