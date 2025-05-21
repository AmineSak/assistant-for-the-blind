# Basic test with default webcam
python environment_description.py --half_precision --display

# Full system with all features
python environment_description.py --half_precision --display --camera_id 0 \
    --width 640 --height 480 --beam_size 3 --cooldown 3.0 \
    --tts_engine coqui