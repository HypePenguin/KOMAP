@echo off
cd /d "C:\Users\SSAFY\Desktop\KOMAP"
"C:\Users\SSAFY\Desktop\KOMAP\.venv\Scripts\python.exe" -u -B -X utf8 "C:\Users\SSAFY\Desktop\KOMAP\src\priority_style89.py" --resume 1>>"C:\Users\SSAFY\Desktop\KOMAP\priority_style89_20261015\execution.log" 2>>"C:\Users\SSAFY\Desktop\KOMAP\priority_style89_20261015\execution.stderr.log"
set "KOMAP_PRIORITY_EXIT=%ERRORLEVEL%"
echo %KOMAP_PRIORITY_EXIT% > "C:\Users\SSAFY\Desktop\KOMAP\priority_style89_20261015\resume_native_20261015_1428.exit_code.txt"
exit /b %KOMAP_PRIORITY_EXIT%
