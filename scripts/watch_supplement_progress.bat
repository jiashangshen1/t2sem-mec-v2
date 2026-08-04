@echo off
chcp 65001 >nul
REM Alias: delegates to start_progress_watch.bat (single-instance watcher + browser)
call "%~dp0start_progress_watch.bat"
