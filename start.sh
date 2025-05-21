# Create directory for models
mkdir -p ~/environment_system/models

# Download script to pre-download models
cat > ~/environment_system/download_models.py << 'EOL'
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoModelForVision2Seq
from TTS.api import TTS
import os

# Set cache directory in home folder for better organization
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/environment_system/models/hf_cache')
os.environ['HF_HOME'] = os.path.expanduser('~/environment_system/models/hf_home')

# Download vision model
model_name = "Salesforce/blip-image-captioning-base"
print(f"Downloading {model_name}...")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name)

# Download TTS model
print("Downloading TTS model...")
tts = TTS("tts_models/en/vctk/vits")
print("Model downloads complete!")
EOL

# Run the download script
python ~/environment_system/download_models.py