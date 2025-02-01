# Medical Condition Detection System - Windows Installation Guide

## Prerequisites

- Windows 10 or 11
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Webcam
- Modern web browser (Chrome recommended)
- [Git](https://git-scm.com/download/win)

## Installation Steps

1. **Install Visual Studio Build Tools**:
   - Download [Visual Studio Build Tools 2019](https://visualstudio.microsoft.com/vs/older-downloads/)
   - During installation, select "Desktop development with C++"
   - This is required for OpenCV and other packages

2. **Create and activate Conda environment**:
```cmd
:: Open Anaconda Prompt (from Start Menu)
:: Create new environment
conda create -n medical-detection python=3.8
:: Activate environment
conda activate medical-detection
```

3. **Clone the repository**:
```cmd
git clone <repository-url>
cd gig
```

4. **Install dependencies**:
```cmd
:: Install core packages using conda
conda install -c conda-forge opencv
conda install -c conda-forge tensorflow
conda install flask pillow scipy numpy

:: Create uploads directory
mkdir uploads
```

## Running the Application

1. **Activate environment** (in Anaconda Prompt):
```cmd
conda activate medical-detection
```

2. **Start the application**:
```cmd
python app.py
```

3. **Access the application**:
- Open Chrome or Edge
- Go to `http://localhost:5000`
- Allow camera access when prompted

## Troubleshooting Windows-Specific Issues

1. **ModuleNotFoundError**:
```cmd
:: Try installing packages individually
pip install flask==2.0.1
pip install opencv-python==4.8.1.78
pip install tensorflow==2.9.1
pip install pillow numpy scipy
```

2. **OpenCV Error**:
```cmd
:: Alternative installation
conda remove opencv
pip install opencv-python
```

3. **Port in use error**:
```cmd
:: Check if port 5000 is in use
netstat -ano | findstr :5000
:: Kill the process if needed
taskkill /PID <PID> /F
```

4. **Camera access issues**:
- Check Windows Camera privacy settings
  - Windows Settings → Privacy & Security → Camera
  - Enable "Camera access" and "Allow apps to access your camera"
- Try restarting browser
- Check if camera works in Camera app

## Windows Performance Tips

1. **Improve detection speed**:
- Close unnecessary background applications
- Use dedicated graphics if available
- Ensure good lighting conditions

2. **Camera optimization**:
- Use external webcam if built-in camera quality is poor
- Adjust camera settings in Windows Camera app
- Ensure adequate USB bandwidth if using external camera

3. **Browser settings**:
- Clear browser cache if video feed is laggy
- Disable hardware acceleration if experiencing glitches
- Use Chrome or Edge for best performance

## Development on Windows

To modify the code:
1. Use VS Code or PyCharm (recommended IDEs)
2. Install Python extension for VS Code
3. Select the correct Conda environment
4. Use Git Bash for version control

## Additional Notes

- Keep Anaconda and Python packages updated
- Run as administrator if experiencing permission issues
- Disable Windows Defender if experiencing slow startup
- Use full paths if encountering file access issues
