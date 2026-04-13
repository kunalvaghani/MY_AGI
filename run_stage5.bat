@echo off
setlocal

REM Windows launcher for the Stage 5 local AI assistant.
REM Examples:
REM   run_stage5.bat --mode chat
REM   run_stage5.bat --mode transfer-eval
REM   run_stage5.bat --mode adapt-eval --limit 5
REM   run_stage5.bat --mode fewshot-eval --limit 5
REM   run_stage5.bat --mode reason-eval --limit 5

py -3.11 ai_stage5.py %*
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Stage 5 run exited with code %EXIT_CODE%.
)

exit /b %EXIT_CODE%
