@echo off
setlocal

REM Windows launcher for the Stage 3 local AI assistant.
REM Examples:
REM   run_stage3.bat --mode chat
REM   run_stage3.bat --mode transfer-eval
REM   run_stage3.bat --mode adapt-eval --limit 5

py -3.11 ai_stage3.py %*
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Stage 3 run exited with code %EXIT_CODE%.
)

exit /b %EXIT_CODE%
