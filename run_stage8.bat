@echo off
setlocal

REM Windows launcher for the Stage 8 local AI assistant.
REM Examples:
REM   run_stage8.bat --mode chat
REM   run_stage8.bat --mode transfer-eval
REM   run_stage8.bat --mode adapt-eval --limit 5
REM   run_stage8.bat --mode fewshot-eval --limit 5
REM   run_stage8.bat --mode reason-eval --limit 5
REM   run_stage8.bat --mode commonsense-eval --limit 5
REM   run_stage8.bat --mode abstract-eval --limit 5
REM   run_stage8.bat --mode causal-eval --limit 5

py -3.11 ai_stage8.py %*
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Stage 8 run exited with code %EXIT_CODE%.
)

exit /b %EXIT_CODE%
