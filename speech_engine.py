import pyttsx3
import platform
import os

def is_wsl():
    """Check if running under Windows Subsystem for Linux"""
    return 'microsoft-standard' in platform.release().lower() or \
           os.path.exists('/proc/version') and 'microsoft' in open('/proc/version').read().lower()

def speak_disease(message: str):
    """
    Function to convert text to speech using pyttsx3
    Args:
        message (str): The message to be spoken
    """
    try:
        # For WSL, try to use espeak-ng directly if pyttsx3 fails
        if is_wsl():
            try:
                os.system(f'espeak-ng "{message}"')
                return
            except:
                pass
        
        # Regular pyttsx3 approach
        engine = pyttsx3.init()
        if platform.system() == 'Windows':
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)
        
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        engine.say(message)
        engine.runAndWait()
        
    except Exception as e:
        if is_wsl():
            print("WSL Error: Try running these commands:")
            print("sudo apt-get update")
            print("sudo apt-get install -y espeak espeak-ng")
            print("sudo apt-get install -y alsa-utils")
            print("sudo apt-get install -y libasound2-dev")
        elif platform.system() == 'Windows':
            print("Error: Make sure Windows Text-to-Speech is enabled in Windows settings")
        else:
            print("Error: Make sure espeak is installed (sudo apt-get install espeak)")
        print(f"Detailed error: {str(e)}")
        
        # Fallback: At least print the message
        print(f"Text that would have been spoken: {message}")

if __name__ == "__main__":
    speak_disease("Testing speech engine")
