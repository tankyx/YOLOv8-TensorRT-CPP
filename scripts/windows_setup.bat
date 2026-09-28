@echo off
rem Convenience wrapper: double-click, or run from cmd.exe.
rem All arguments are forwarded to windows_setup.ps1, e.g.:
rem   scripts\windows_setup.bat -TensorRtZip "%USERPROFILE%\Downloads\TensorRT-10.9.0.34.Windows.win10.cuda-12.9.zip"
rem   scripts\windows_setup.bat -DryRun
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows_setup.ps1" %*
set RC=%ERRORLEVEL%
if not "%RC%"=="0" echo.
if not "%RC%"=="0" echo windows_setup.ps1 exited with code %RC%
exit /b %RC%
