"""의존성 설치 프로그램, opencv-python-headless가 설치되어야 하는데 opencv-python으로 설치되는 에러 해결"""
import subprocess
import sys
 
 
def pip(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
 
 
pip("install", "-r", "requirements.txt")
pip("uninstall", "-y", "opencv-python", "opencv-python-headless")
pip("install", "opencv-python-headless")