@echo off
setlocal

REM Windows launcher for the Stage 10 local AI assistant.
REM Examples:
REM   run_stage10.bat --mode chat
REM   run_stage10.bat --mode transfer-eval
REM   run_stage10.bat --mode adapt-eval --limit 5
REM   run_stage10.bat --mode fewshot-eval --limit 5
REM   run_stage10.bat --mode reason-eval --limit 5
REM   run_stage10.bat --mode commonsense-eval --limit 5
REM   run_stage10.bat --mode abstract-eval --limit 5
REM   run_stage10.bat --mode causal-eval --limit 5
REM   run_stage10.bat --mode planning-eval --limit 5
REM   run_stage10.bat --mode goal-eval --limit 5

py -3.11 ai_stage10.py %*
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Stage 10 run exited with code %EXIT_CODE%.
)

exit /b %EXIT_CODE%
