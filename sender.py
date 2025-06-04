"""
Simplified real-time environment description system using BLIP vision model
Sends descriptions via UDP to another PC for text-to-speech processing
Optimized for Jetson Nano
"""

import json
import socket
import threading
import time

import cv2
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class RealTimeDescriberUDP:
    def __init__(self, target_ip="192.168.1.100", target_port=12345):
        # Initialize state variables first
        self.running = False
        self.current_frame = None
        self.last_description = ""
        self.last_description_time = 0
        self.description_cooldown = 5.0  # seconds between descriptions

        # UDP configuration
        self.target_ip = target_ip
        self.target_port = target_port
        self.sock = None

        # Now setup components
        self.setup_udp()
        self.setup_model()
        self.setup_camera()

    def setup_udp(self):
        """Initialize UDP socket"""
        print(f"Setting up UDP socket to {self.target_ip}:{self.target_port}")
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Test connection
            test_msg = json.dumps(
                {"type": "test", "message": "Connection test from Jetson Nano"}
            )
            self.sock.sendto(
                test_msg.encode("utf-8"), (self.target_ip, self.target_port)
            )
            print("UDP socket ready")
        except Exception as e:
            print(f"UDP setup failed: {e}")
            print("Make sure the target PC is reachable and the receiver is running")

    def setup_model(self):
        """Initialize BLIP model"""
        print("Loading BLIP model...")
        model_name = "Salesforce/blip-image-captioning-base"

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

        # Use GPU if available on Jetson Nano
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on {self.device}")

    def setup_camera(self):
        """Initialize camera"""
        print("Setting up camera...")
        # Try different camera indices for Jetson Nano
        for camera_index in [0, 1]:
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                break

        if not self.cap.isOpened():
            # Try with gstreamer pipeline for Jetson Nano CSI camera
            gst_pipeline = (
                "nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
                "nvvidconv ! video/x-raw, format=BGRx ! "
                "videoconvert ! video/x-raw, format=BGR ! appsink"
            )
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("Camera ready")
        else:
            raise RuntimeError("Could not open camera")

    def capture_frame(self):
        """Capture a single frame from camera"""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
        return ret

    def generate_description(self, frame):
        """Generate description for the current frame"""
        try:
            # Convert frame to PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Process with BLIP using inference mode
            with torch.inference_mode():
                # Process image
                inputs = self.processor(
                    images=pil_image,
                    return_tensors="pt",
                    truncation=True,
                ).to(self.device)

                # Generate description with optimized parameters for Jetson Nano
                outputs = self.model.generate(
                    **inputs,
                    max_length=30,  # Reduced for faster processing
                    num_beams=3,  # Reduced for faster processing
                    do_sample=False,
                    early_stopping=True,
                )

                description = self.processor.decode(
                    outputs[0], skip_special_tokens=True
                )

            return description.strip()

        except Exception as e:
            print(f"Error generating description: {e}")
            return None

    def is_description_different(self, new_desc):
        """Check if new description is significantly different from the last one"""
        if not self.last_description:
            return True

        # Simple word-based difference check
        old_words = set(self.last_description.lower().split())
        new_words = set(new_desc.lower().split())

        # Calculate overlap ratio
        overlap = len(old_words.intersection(new_words))
        total_unique = len(old_words.union(new_words))

        if total_unique == 0:
            return False

        similarity = overlap / total_unique
        return similarity < 0.7  # Consider different if less than 70% similar

    def send_description_udp(self, description):
        """Send description via UDP"""
        if not self.sock:
            print("UDP socket not available")
            return False

        try:
            # Create message with timestamp and description
            message = {
                "type": "description",
                "text": description,
                "timestamp": time.time(),
                "source": "jetson_nano",
            }

            # Convert to JSON and send
            json_msg = json.dumps(message)
            self.sock.sendto(
                json_msg.encode("utf-8"), (self.target_ip, self.target_port)
            )
            print(f"Sent via UDP: {description}")
            return True

        except Exception as e:
            print(f"Error sending UDP message: {e}")
            return False

    def process_and_describe(self):
        """Main processing loop for generating descriptions"""
        while self.running:
            current_time = time.time()

            # Check cooldown period
            if current_time - self.last_description_time < self.description_cooldown:
                time.sleep(1)
                continue

            if self.current_frame is not None:
                # Generate description
                description = self.generate_description(self.current_frame)

                if description and self.is_description_different(description):
                    print(f"Description: {description}")

                    # Update tracking variables
                    self.last_description = description
                    self.last_description_time = current_time

                    # Send description via UDP
                    self.send_description_udp(description)

            time.sleep(0.1)

    def display_video(self):
        """Display video feed with overlay text"""
        while self.running:
            if not self.capture_frame():
                print("Failed to capture frame")
                time.sleep(0.1)
                continue

            # Create display frame
            display_frame = self.current_frame.copy()

            # Add description text overlay
            if self.last_description:
                # Wrap text for display
                words = self.last_description.split()
                lines = []
                current_line = []
                max_chars_per_line = 40

                for word in words:
                    test_line = " ".join(current_line + [word])
                    if len(test_line) <= max_chars_per_line:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(" ".join(current_line))
                        current_line = [word]

                if current_line:
                    lines.append(" ".join(current_line))

                # Draw text lines
                y_offset = 25
                for line in lines:
                    cv2.putText(
                        display_frame,
                        line,
                        (5, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    y_offset += 25

            # Add UDP status indicator
            status_text = f"UDP -> {self.target_ip}:{self.target_port}"
            cv2.putText(
                display_frame,
                status_text,
                (5, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

            # Show frame
            cv2.imshow("Jetson Nano - Environment Description", display_frame)

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                self.stop()
                break

    def start(self):
        """Start the real-time description system"""
        print("Starting Jetson Nano Environment Describer with UDP...")
        print(f"Sending descriptions to {self.target_ip}:{self.target_port}")
        print("Press 'q' or ESC to quit")

        self.running = True

        # Start description processing in background thread
        self.process_thread = threading.Thread(
            target=self.process_and_describe, daemon=True
        )
        self.process_thread.start()

        # Run video display in main thread
        self.display_video()

    def stop(self):
        """Stop the system"""
        print("Stopping...")
        self.running = False

        # Send stop message
        if self.sock:
            try:
                stop_msg = json.dumps({"type": "stop", "source": "jetson_nano"})
                self.sock.sendto(
                    stop_msg.encode("utf-8"), (self.target_ip, self.target_port)
                )
                self.sock.close()
            except Exception as e:
                print(f"Error closing UDP socket: {e}")

        if hasattr(self, "cap"):
            self.cap.release()

        cv2.destroyAllWindows()
        print("Stopped")


def main():
    """Main function"""
    print("Jetson Nano Real-time Environment Description System with UDP")
    print("=" * 55)

    # Get target IP from user or use default
    target_ip = input("Enter target PC IP address (default: 192.168.1.100): ").strip()
    if not target_ip:
        target_ip = "192.168.1.100"

    target_port = input("Enter target port (default: 12345): ").strip()
    if not target_port:
        target_port = 12345
    else:
        target_port = int(target_port)

    try:
        describer = RealTimeDescriberUDP(target_ip, target_port)
        describer.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Shutting down...")


if __name__ == "__main__":
    main()
