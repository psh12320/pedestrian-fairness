@echo off
setlocal
cd /d "%~dp0"
if exist ".local\sumo-1.27.1\tools\traci\__init__.py" (
    set "PYTHONPATH=%~dp0.local\sumo-1.27.1\tools;%PYTHONPATH%"
    set "SUMO_HOME="
)
echo Opening fixed-time and pedestrian-aware IPPO with the same traffic demand.
echo The first launch prepares the two runs before opening their SUMO windows.
python -m src.traffic_control.viewer --compare --seed 6001 --pedestrian-count 120 %*
if errorlevel 1 (
    echo.
    echo The comparison could not start. Check the message above and README.md.
    pause
)
endlocal
