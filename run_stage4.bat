@echo off
setlocal

REM Windows launcher for the Stage 4 local AI assistant.
REM Examples:
REM   run_stage4.bat --mode chat
REM   run_stage4.bat --mode transfer-eval
REM   run_stage4.bat --mode adapt-eval --limit 5
REM   run_stage4.bat --mode fewshot-eval --limit 5

py -3.11 ai_stage4.py %*
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Stage 4 run exited with code %EXIT_CODE%.
)

exit /b %EXIT_CODE%
