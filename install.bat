@echo off
echo Installing dependencies...
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
echo Done!
