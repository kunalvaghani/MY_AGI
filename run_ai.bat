@echo off
setlocal

REM Always run from the folder that contains this batch file.
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Forward any command-line arguments, for example:
REM run_ai.bat --model llama3.2:latest
py -3.11 ai_stage1.py %*

set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%