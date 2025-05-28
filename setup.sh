# Update package lists
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-dev libportaudio2 libasound-dev \
    python3-opencv python3-pygame

# Create and activate virtual environment (recommended)
python3 -m pip install virtualenv
python3 -m virtualenv ~/env_desc
source ~/env_desc/bin/activate

# Install PyTorch for Jetson
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

# Install additional Python packages
pip install transformers numpy opencv-python pygame pyttsx3

# Install TTS (might take a while)
pip install TTS