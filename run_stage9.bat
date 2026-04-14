@echo off
setlocal

REM Windows launcher for the Stage 9 local AI assistant.
REM Examples:
REM   run_stage9.bat --mode chat
REM   run_stage9.bat --mode transfer-eval
REM   run_stage9.bat --mode adapt-eval --limit 5
REM   run_stage9.bat --mode fewshot-eval --limit 5
REM   run_stage9.bat --mode reason-eval --limit 5
REM   run_stage9.bat --mode commonsense-eval --limit 5
REM   run_stage9.bat --mode abstract-eval --limit 5
REM   run_stage9.bat --mode causal-eval --limit 5
REM   run_stage9.bat --mode planning-eval --limit 5

py -3.11 ai_stage9.py %*
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Stage 9 run exited with code %EXIT_CODE%.
)

exit /b %EXIT_CODE%
