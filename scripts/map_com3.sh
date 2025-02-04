#!/bin/bash
# List available USB devices from Windows in WSL
echo "Listing Windows USB devices available to WSL:"
usbipd wsl list

# Replace <your_busid> with the bus ID corresponding to COM3 (e.g., 1-1.3)
BUSID="<your_busid>"
if [ "$BUSID" = "<your_busid>" ]; then
    echo "Please update the script with your actual bus ID for COM3."
    exit 1
fi

echo "Attaching device with bus ID $BUSID ..."
sudo usbipd wsl attach --busid "$BUSID"

# Optionally, create a symlink if needed (adjust ttyS number if necessary)
DEVICE=$(ls /dev/ttyS* | head -n 1)
echo "Device attached as: $DEVICE"
# Uncomment the following line if your app expects /dev/ttyACM0
# sudo ln -sf $DEVICE /dev/ttyACM0

echo "Mapping complete. Verify with: ls /dev/ttyS*"
