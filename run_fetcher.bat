@echo off
setlocal
pushd "%~dp0"
REM Launch the fetcher deterministically, linear to the absolute last page
py -3 original.py --deterministic --chunk-size 1000000 --last-page "904625697166532776746648320380374280100293470930272690489102837043110636675" --processes 32
popd
endlocal
