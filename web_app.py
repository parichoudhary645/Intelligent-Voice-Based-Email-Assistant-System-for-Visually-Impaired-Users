from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask_socketio import SocketIO, emit
import speech_recognition as sr
import threading
import queue
import time
import os
import subprocess
from pathlib import Path
import logging
from voice_auth import VoiceAuthenticator
from gmail_service import send_email_via_gmail, get_service
from semantic_search import run_semantic_search, get_most_recent_email
import multiprocessing as mp
import cv2
from gesture_process import detect_gesture  # Import the detect_gesture function
import traceback

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize speech recognizer with same settings as app.py
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = True
recognizer.energy_threshold = 200  # Lower threshold for better sensitivity
recognizer.pause_threshold = 2.0    # Longer pause threshold
recognizer.phrase_threshold = 0.1   # More sensitive phrase detection
recognizer.non_speaking_duration = 1.5  # Longer non-speaking duration

# Global variables
audio_queue = queue.Queue()
is_listening = False
current_state = "welcome"
current_username = None
authenticator = VoiceAuthenticator()
max_retries = 3
speech_rate = 180  # Reduced from 220 to make voice commands slower
microphone = None  # Global microphone instance

# Gesture recognition variables
gesture_queue = None
gesture_process = None
stop_gesture_event = None

def gesture_recognition_process(queue, stop_event):
    """Process for running gesture recognition"""
    try:
        print("\n🎥 Starting gesture recognition process...")
        
        # Import MediaPipe here in the child process
        import mediapipe as mp
        
        # Initialize MediaPipe first
        try:
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.1
            )
            mp_draw = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            print("✅ MediaPipe Hands initialized")
        except Exception as e:
            print(f"❌ Failed to initialize MediaPipe: {e}")
            return
        
        # List and select camera
        print("\nScanning for available cameras...")
        available_cameras = []
        for device_id in range(10):  # Check more camera indices
            try:
                cap = cv2.VideoCapture(device_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Get camera properties
                        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        camera_info = {
                            'device_id': device_id,
                            'resolution': f"{int(width)}x{int(height)}",
                            'fps': fps,
                            'frame_shape': frame.shape
                        }
                        available_cameras.append(camera_info)
                        print(f"✅ Found camera {device_id}: {camera_info}")
                    cap.release()
            except Exception as e:
                print(f"Failed to check camera device {device_id}: {e}")
        
        if not available_cameras:
            print("❌ No cameras found")
            print("Please ensure camera permissions are granted in System Preferences > Security & Privacy > Privacy > Camera")
            return
        
        # Try to use the first available camera
        selected_camera = available_cameras[0]
        print(f"\nSelected camera: {selected_camera}")
        
        try:
            cap = cv2.VideoCapture(selected_camera['device_id'])
            if not cap.isOpened():
                print(f"❌ Failed to open camera device {selected_camera['device_id']}")
                return
            
            # Try to read a test frame
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("❌ Failed to read test frame")
                return
                
            print(f"✅ Camera opened successfully:")
            print(f"  - Device ID: {selected_camera['device_id']}")
            print(f"  - Resolution: {selected_camera['resolution']}")
            print(f"  - FPS: {selected_camera['fps']}")
            print(f"  - Frame shape: {test_frame.shape}")
            
            # Set camera properties for optimal performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            return
        
        # Wait for camera to initialize
        time.sleep(2)
        
        # Create window with explicit flags
        window_name = 'Gesture Recognition'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        cv2.moveWindow(window_name, 100, 100)
        
        print("\n👋 Ready for gestures!")
        print("Available gestures:")
        print("1. 👍 Thumbs up  -> send email")
        print("2. ✋ Palm       -> cancel email")
        
        last_gesture = None
        gesture_counter = 0
        GESTURE_THRESHOLD = 1  # Reduced threshold for faster detection
        frame_count = 0
        last_frame_time = time.time()
        gesture_detected = False
        gesture_start_time = 0
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("❌ Camera read error - trying to reconnect...")
                time.sleep(1)
                continue
            
            # Calculate and print FPS
            current_time = time.time()
            fps = 1 / (current_time - last_frame_time)
            last_frame_time = current_time
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print status every 30 frames
                print(f"📸 Processing frame {frame_count} at {fps:.1f} FPS")
            
            # Resize frame if needed
            if frame.shape[1] > 640:  # If width is larger than 640
                scale = 640 / frame.shape[1]
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Draw hand landmarks with improved visibility
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks with custom style
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Detect gesture
                    current_gesture = detect_gesture(hand_landmarks)
                    
                    # Gesture smoothing with improved stability
                    if current_gesture == last_gesture:
                        gesture_counter += 1
                        if gesture_counter == 1:
                            print("🔍 Potential gesture detected, hold steady...")
                            gesture_start_time = time.time()
                    else:
                        gesture_counter = max(0, gesture_counter - 1)  # Gradual decrease
                        gesture_detected = False
                    
                    if gesture_counter >= GESTURE_THRESHOLD and current_gesture:
                        if not gesture_detected:
                            print(f"✨ Gesture detected: {current_gesture}")
                            queue.put(("gesture", current_gesture))
                            gesture_detected = True
                            # Add a small delay after gesture detection
                            time.sleep(0.2)  # Reduced delay for faster response
                    
                    last_gesture = current_gesture
                    
                    # Draw gesture progress bar with improved visibility
                    if gesture_counter > 0:
                        progress = min(1.0, gesture_counter / GESTURE_THRESHOLD)
                        bar_width = int(frame.shape[1] * progress)
                        # Draw background
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], 20), (0, 0, 0), -1)
                        # Draw progress
                        cv2.rectangle(frame, (0, 0), (bar_width, 20), (0, 255, 0), -1)
                        # Add percentage text
                        cv2.putText(frame, f"{int(progress * 100)}%", (10, 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Show gesture name when detected
                    if gesture_detected:
                        gesture_text = "THUMBS UP - Sending" if current_gesture == "send" else "PALM - Cancelling"
                        # Draw background for text
                        cv2.rectangle(frame, (0, 30), (400, 60), (0, 0, 0), -1)
                        cv2.putText(frame, gesture_text, (10, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                gesture_counter = max(0, gesture_counter - 1)  # Gradual decrease
                gesture_detected = False
            
            # Add gesture guide text with improved visibility
            cv2.rectangle(frame, (0, frame.shape[0] - 60), (400, frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, "Show gesture and hold steady", (10, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Thumbs up = Send  |  Palm = Cancel", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add frame counter and FPS
            cv2.putText(frame, f"Frame: {frame_count} | FPS: {fps:.1f}", (10, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow(window_name, cv2.flip(frame, 1))
            
            # Check for ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("👋 Gesture recognition stopped by user")
                break
    
    except Exception as e:
        print(f"❌ Gesture recognition error: {e}")
        print(f"Error details: {str(e)}")
        traceback.print_exc()
    
    finally:
        print("\n🎥 Cleaning up...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        # Extra cleanup for macOS
        for i in range(4):
            cv2.waitKey(1)
        if 'hands' in locals():
            hands.close()

def start_gesture_recognition():
    """Start the gesture recognition process"""
    global gesture_queue, gesture_process, stop_gesture_event
    
    if gesture_process is None:
        try:
            # Create new queue and event
            gesture_queue = mp.Queue()
            stop_gesture_event = mp.Event()
            
            # Start the process with proper error handling
            gesture_process = mp.Process(
                target=gesture_recognition_process,
                args=(gesture_queue, stop_gesture_event),
                daemon=True
            )
            
            # Start the process
            gesture_process.start()
            
            # Wait for process to initialize
            time.sleep(3)
            
            if not gesture_process.is_alive():
                raise Exception("Gesture recognition process failed to start")
            
            print("✅ Gesture recognition process started")
            socketio.emit('gesture_status', {'status': 'started'})
            
        except Exception as e:
            print(f"❌ Error starting gesture recognition: {e}")
            speak("I'm having trouble accessing the camera. Please ensure camera permissions are granted in System Preferences.")
            socketio.emit('gesture_status', {'status': 'error', 'message': str(e)})
            # Clean up if process was partially started
            if gesture_process is not None:
                gesture_process.terminate()
                gesture_process = None
            if gesture_queue is not None:
                gesture_queue = None
            if stop_gesture_event is not None:
                stop_gesture_event = None

def stop_gesture_recognition():
    """Stop the gesture recognition process"""
    global gesture_process, stop_gesture_event
    
    if gesture_process is not None:
        try:
            # Signal the process to stop
            stop_gesture_event.set()
            
            # Wait for process to finish
            gesture_process.join(timeout=5)
            
            # Force terminate if still running
            if gesture_process.is_alive():
                gesture_process.terminate()
                gesture_process.join(timeout=1)
            
            gesture_process = None
            print("✅ Gesture recognition process stopped")
            socketio.emit('gesture_status', {'status': 'stopped'})
            
        except Exception as e:
            print(f"❌ Error stopping gesture recognition: {e}")
            if gesture_process is not None:
                gesture_process.terminate()
                gesture_process = None

def check_gesture_command():
    """Check for gesture commands in the queue"""
    global gesture_queue
    
    if gesture_queue is not None:
        try:
            # Check for any pending gestures without blocking
            if not gesture_queue.empty():
                msg = gesture_queue.get_nowait()
                if isinstance(msg, tuple) and msg[0] == "gesture":
                    gesture = msg[1]
                    print(f"🎯 Gesture command received: {gesture}")
                    # Immediately emit the command
                    socketio.emit('gesture_command', {'command': gesture})
                    # Return the gesture immediately
                    return gesture
                elif isinstance(msg, tuple) and msg[0] == "error":
                    print(f"❌ Gesture error: {msg[1]}")
                    return None
        except Exception as e:
            print(f"Error checking gesture queue: {e}")
    return None

def speak(text):
    """Speak the given text using macOS say command with faster rate"""
    try:
        print(f"[TTS] {text}")
        subprocess.run(['say', '-r', str(speech_rate), text], check=True)
        time.sleep(0.5)  # Add delay after each speech
        socketio.emit('speak', {'text': text})
        logging.info(f"Speaking: {text}")
    except Exception as e:
        print(f"Error in speak function: {e}")
        logging.error(f"Error in speak function: {e}")

def initialize_microphone():
    """Initialize and return a persistent microphone instance"""
    global microphone
    if microphone is None:
        try:
            microphone = sr.Microphone()
            with microphone as source:
                print("[System] Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=3)  # Longer adjustment time
                print("[System] Microphone initialized and calibrated")
        except Exception as e:
            print(f"[System] Error initializing microphone: {e}")
            return None
    return microphone

def recognize_speech(timeout=5, phrase_time_limit=None, is_username=False):
    """Recognize speech using Google Speech Recognition with better feedback"""
    global microphone
    
    # Use different settings for username input
    if is_username:
        timeout = 30  # Increased timeout for username
        phrase_time_limit = None  # No phrase time limit for username
        print("[System] I'm waiting for your username. Please speak clearly.")
        speak("I'm waiting for your username. Please speak clearly.")
        time.sleep(2)  # Give user more time to prepare
    
    print("\n🎤 Listening...")
    
    try:
        # Use the persistent microphone instance
        if microphone is None:
            microphone = initialize_microphone()
            if microphone is None:
                speak("There was an error with the microphone. Please refresh the page.")
                return None
            
        with microphone as source:
            # Adjust for ambient noise with longer duration for username
            if is_username:
                print("[System] Recalibrating for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=3)
                print("[System] Ready to listen for username")
            
            print("[System] Microphone is active. Please speak now...")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        
        print("[System] Processing speech...")
        
        try:
            text = recognizer.recognize_google(audio)
            print(f"[You said] {text}")
            
            # Additional validation for username
            if is_username:
                # Remove common system phrases that might be misinterpreted
                text = text.lower()
                invalid_phrases = [
                    "welcome to voice email system",
                    "voice email system",
                    "welcome",
                    "system",
                    "email",
                    "voice"
                ]
                
                # Check if the input is a system phrase
                if any(phrase in text for phrase in invalid_phrases):
                    print("[System] That appears to be a system message. Please say your username.")
                    speak("That appears to be a system message. Please say your username.")
                    return None
                
                # Ensure username is not too long
                if len(text) > 20:
                    print("[System] Username is too long. Please use a shorter username.")
                    speak("Username is too long. Please use a shorter username.")
                    return None
                
                # Ensure username is not too short
                if len(text) < 2:
                    print("[System] Username is too short. Please use a longer username.")
                    speak("Username is too short. Please use a longer username.")
                    return None
            
            return text.lower()
            
        except sr.UnknownValueError:
            if is_username:
                speak("I didn't catch your username. Please try again.")
            else:
                speak("I didn't catch that. Please try again.")
            return None
        except sr.RequestError as e:
            print(f"[System] Could not request results; {e}")
            speak("I'm having trouble connecting to the speech service. Please try again.")
            return None
            
    except sr.WaitTimeoutError:
        if is_username:
            speak("I didn't hear anything. Please say your username.")
        else:
            speak("I didn't hear anything. Please try again.")
        return None
    except Exception as e:
        print(f"[System] Error in speech recognition: {e}")
        speak("There was an error with the speech recognition. Please try again.")
        return None

def get_email_details():
    """Get email details with voice confirmation"""
    # Get recipient email with longer timeout
    for _ in range(max_retries):
        speak("Please prepare to say the recipient's email address.")
        time.sleep(1)  # Give user time to prepare
        speak("Listening now...")
        
        email = recognize_speech(timeout=10, phrase_time_limit=7)
        if not email:
            speak("I didn't hear an email address. Let's try again.")
            continue
            
        email = email.replace(" at ", "@").replace(" dot ", ".").replace(" ", "")
        speak(f"I heard the email address as: {email}")
        speak("To confirm this email address, say 'confirm'. To try again, say 'cancel'.")
        
        confirmation = recognize_speech(timeout=5, phrase_time_limit=2)
        if confirmation:
            if 'confirm' in confirmation.lower():
                break
            elif 'cancel' in confirmation.lower():
                if _ < max_retries - 1:
                    speak("Let's try again.")
                continue
            else:
                speak("Please say either confirm or cancel.")
                continue
        else:
            speak("I didn't hear your response. Please say confirm or cancel clearly.")
            continue
    
    else:  # If we've exhausted all retries
        speak("Too many unsuccessful attempts. Cancelling email composition.")
        return None, None, None

    # Get subject
    speak("Please dictate the subject of your email.")
    subject = recognize_speech(timeout=7)
    if not subject:
        return None, None, None

    # Get body
    speak("Now please dictate the content of your email.")
    body = recognize_speech(phrase_time_limit=30)
    if not body:
        return None, None, None

    return email, subject, body

def handle_voice_enrollment(username):
    """Handle voice enrollment process"""
    speak(f"Let's create your voice profile, {username}. You'll need to say 5 different phrases.")
    time.sleep(1)  # Give user time to prepare
    
    try:
        # Start enrollment process
        if authenticator.enroll_user(username):
            # Don't speak here as enroll_user already speaks
            return True
        else:
            speak("Failed to create voice profile. Please try again.")
            return False
    except Exception as e:
        print(f"Error in voice enrollment: {e}")
        speak("There was an error during voice enrollment. Please try again.")
        return False

def handle_voice_input():
    """Handle voice input in a separate thread"""
    global is_listening, current_state, current_username
    
    while is_listening:
        command = recognize_speech()
        if command:
            socketio.emit('voice_command', {'command': command})
            
            if current_state == "welcome":
                # Validate username input
                if command:
                    # Check for system messages
                    invalid_phrases = [
                        "welcome to voice email system",
                        "voice email system",
                        "welcome",
                        "system",
                        "email",
                        "voice"
                    ]
                    
                    if any(phrase in command.lower() for phrase in invalid_phrases):
                        speak("That appears to be a system message. Please say your username.")
                        continue
                        
                    # Validate username length
                    if len(command) < 2 or len(command) > 20:
                        speak("Please use a username between 2 and 20 characters.")
                        continue
                    
                    current_username = command
                    profile_path = authenticator.voice_dir / f"{current_username}_profile.pkl"
                    
                    # Add delay before next prompt
                    time.sleep(1)
                    
                    if profile_path.exists():
                        current_state = "auth"
                        socketio.emit('state_change', {'state': 'auth', 'username': command})
                        speak(f"Welcome back {command}!")
                        time.sleep(1)  # Pause between messages
                        speak("Please verify your voice by saying: Hello this is my voice")
                    else:
                        current_state = "enroll"
                        socketio.emit('state_change', {'state': 'enroll', 'username': command})
                        time.sleep(1)  # Pause before starting enrollment
                        if handle_voice_enrollment(command):
                            current_state = "main_menu"
                            socketio.emit('state_change', {'state': 'main_menu'})
                            speak("Voice profile created successfully!")
                            time.sleep(1)  # Pause between messages
                            speak("What would you like to do? Say compose to write an email, search to find emails, recent to see your latest email, or exit to quit.")
                        else:
                            current_state = "welcome"
                            socketio.emit('state_change', {'state': 'welcome'})
                            speak("Voice enrollment failed. Please try again.")
                            time.sleep(1)  # Pause before asking username again
                            speak("Please say your username when you're ready.")
                    
            elif current_state == "auth":
                if authenticator.verify_user(current_username):
                    current_state = "main_menu"
                    socketio.emit('state_change', {'state': 'main_menu'})
                    speak("Voice authentication successful!")
                    speak("What would you like to do? Say compose to write an email, search to find emails, recent to see your latest email, or exit to quit.")
                else:
                    socketio.emit('auth_failed')
                    speak("Voice authentication failed. Please try again.")
                    
            elif current_state == "main_menu":
                if command == "compose":
                    current_state = "compose"
                    socketio.emit('state_change', {'state': 'compose'})
                    handle_compose_email()
                elif command == "search":
                    current_state = "search"
                    socketio.emit('state_change', {'state': 'search'})
                    handle_email_search()
                elif command == "recent":
                    current_state = "recent"
                    socketio.emit('state_change', {'state': 'recent'})
                    handle_recent_email()
                elif command == "exit":
                    current_state = "welcome"
                    current_username = None
                    socketio.emit('state_change', {'state': 'welcome'})
                    speak("Goodbye! Thank you for using Voice Email System.")
                else:
                    speak("I didn't understand that command. Please say compose, search, recent, or exit.")

def handle_compose_email():
    """Handle email composition with gesture support"""
    global current_state
    
    # Get email details first without gesture recognition
    email, subject, body = get_email_details()
    
    if email and subject and body:
        # Start gesture recognition only for confirmation
        speak("Here's your email for review:")
        speak(f"To: {email}")
        speak(f"Subject: {subject}")
        speak(f"Content: {body}")
        time.sleep(1)  # Pause before starting gesture recognition
        
        speak("You can either say 'confirm' or show a thumbs up to send the email.")
        speak("Or say 'cancel' or show your palm to cancel.")
        
        # Start gesture recognition before waiting for voice command
        start_gesture_recognition()
        time.sleep(3)  # Give more time for camera initialization
        
        # Wait for either voice command or gesture
        start_time = time.time()
        confirmation_received = False
        gesture_initialized = False
        
        while time.time() - start_time < 30 and not confirmation_received:  # 30 second timeout
            # Check if gesture recognition is properly initialized
            if not gesture_initialized:
                time.sleep(2)  # Give more time for initialization
                gesture_initialized = True
                continue
            
            # Check for gesture command with higher priority
            gesture = check_gesture_command()
            if gesture == "send":
                try:
                    print("📧 Sending email immediately...")
                    # Stop gesture recognition before sending email
                    stop_gesture_recognition()
                    # Send the email
                    send_email_via_gmail(email, subject, body)
                    speak("Email sent successfully!")
                    confirmation_received = True
                    break  # Exit immediately after sending
                except Exception as e:
                    print(f"Error sending email: {e}")
                    speak("Failed to send email. Please try again.")
                    break
            elif gesture == "cancel":
                print("❌ Email cancelled by gesture")
                # Stop gesture recognition before cancelling
                stop_gesture_recognition()
                speak("Email cancelled.")
                confirmation_received = True
                break  # Exit immediately after cancelling
            
            # Only check for voice command if no gesture was detected
            if not confirmation_received:
                voice_command = recognize_speech(timeout=1, phrase_time_limit=2)
                if voice_command:
                    if 'confirm' in voice_command.lower():
                        try:
                            print("📧 Sending email...")
                            # Stop gesture recognition before sending email
                            stop_gesture_recognition()
                            send_email_via_gmail(email, subject, body)
                            speak("Email sent successfully!")
                            confirmation_received = True
                            break
                        except Exception as e:
                            print(f"Error sending email: {e}")
                            speak("Failed to send email. Please try again.")
                            break
                    elif 'cancel' in voice_command.lower():
                        print("❌ Email cancelled by voice")
                        # Stop gesture recognition before cancelling
                        stop_gesture_recognition()
                        speak("Email cancelled.")
                        confirmation_received = True
                        break
            
            time.sleep(0.1)  # Small delay to prevent CPU overuse
        
        if not confirmation_received:
            speak("No confirmation received. Cancelling email.")
            stop_gesture_recognition()
        
        # Give time for cleanup
        time.sleep(1)
        speak("Returning to main menu.")
    else:
        speak("Email composition cancelled.")
    
    current_state = "main_menu"
    socketio.emit('state_change', {'state': 'main_menu'})
    speak("What would you like to do? Say compose to write an email, search to find emails, recent to see your latest email, or exit to quit.")
    return False

def handle_email_search():
    """Handle email search process"""
    global current_state
    
    speak("What kind of emails would you like to search for?")
    query = recognize_speech()
    
    if not query:
        speak("No search query provided.")
        current_state = "main_menu"
        socketio.emit('state_change', {'state': 'main_menu'})
        speak("What would you like to do? Say compose to write an email, search to find emails, recent to see your latest email, or exit to quit.")
        return
    
    try:
        matching_emails = run_semantic_search(query)
        
        if matching_emails:
            speak(f"I found {len(matching_emails)} matching emails. I'll read them to you now.")
            
            for i, match in enumerate(matching_emails, 1):
                email = match['email']
                similarity = match['similarity']
                
                speak(f"Email {i} with {int(similarity * 100)}% relevance:")
                speak(f"From: {email['from']}")
                speak(f"Subject: {email['subject']}")
                speak("Message content:")
                speak(email['body'])
                
                if i < len(matching_emails):
                    speak("Would you like to hear the next email? Say yes or no.")
                    response = recognize_speech(timeout=3)
                    if not response or 'no' in response:
                        break
        else:
            speak("No matching emails found.")
    except Exception as e:
        logging.error(f"Error searching emails: {e}")
        speak("Sorry, I encountered an error while searching emails.")
    
    current_state = "main_menu"
    socketio.emit('state_change', {'state': 'main_menu'})
    speak("What would you like to do? Say compose to write an email, search to find emails, recent to see your latest email, or exit to quit.")

def handle_recent_email():
    """Handle fetching most recent email"""
    global current_state
    
    speak("Fetching your most recent email...")
    try:
        result = get_most_recent_email()
        
        if result:
            formatted_email, email_content = result
            speak("Here is your most recent email:")
            speak(f"From: {email_content['from']}")
            speak(f"Subject: {email_content['subject']}")
            speak("Message content:")
            speak(email_content['body'])
        else:
            speak("No recent emails found.")
    except Exception as e:
        logging.error(f"Error fetching recent email: {e}")
        speak("Sorry, I encountered an error while fetching your recent email.")
    
    current_state = "main_menu"
    socketio.emit('state_change', {'state': 'main_menu'})
    speak("What would you like to do? Say compose to write an email, search to find emails, recent to see your latest email, or exit to quit.")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    global is_listening, microphone
    print("Client connected")
    
    # Initial welcome message without microphone
    speak("Welcome to Voice Email System")
    time.sleep(1)  # Reduced from 3 to 1 second
    
    # Ask for username without microphone
    speak("Please say your username")
    time.sleep(1)  # Wait for prompt to finish
    
    # Now initialize microphone and start listening
    if not is_listening:
        is_listening = True
        microphone = initialize_microphone()
        if microphone is None:
            speak("There was an error with the microphone. Please refresh the page.")
            return
        threading.Thread(target=handle_voice_input, daemon=True).start()

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    global is_listening, microphone
    print("Client disconnected")
    is_listening = False
    stop_gesture_recognition()
    # Clean up microphone
    if microphone is not None:
        microphone = None

@socketio.on('start_listening')
def handle_start_listening():
    """Start voice input handling"""
    global is_listening
    if not is_listening:
        is_listening = True
        threading.Thread(target=handle_voice_input, daemon=True).start()

@socketio.on('stop_listening')
def handle_stop_listening():
    """Stop voice input handling"""
    global is_listening
    is_listening = False
    stop_gesture_recognition()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        if not username:
            speak("Please enter a username")
            return render_template('login.html', error="Please enter a username")
            
        # Check if user exists
        profile_path = Path("voice_auth") / f"{username}_profile.pkl"
        
        if profile_path.exists():
            # Existing user - verify voice
            speak(f"Welcome back {username}. Please verify your voice.")
            if authenticator.verify_user(username):
                speak("Voice verification successful!")
                session['username'] = username
                return redirect(url_for('dashboard'))
            else:
                speak("Voice verification failed. Please try again.")
                return render_template('login.html', error="Voice verification failed")
        else:
            # New user - enroll voice
            speak(f"Welcome {username}. Let's create your voice profile.")
            if authenticator.enroll_user(username):
                speak("Voice profile created successfully!")
                session['username'] = username
                return redirect(url_for('dashboard'))
            else:
                speak("Failed to create voice profile. Please try again.")
                return render_template('login.html', error="Failed to create voice profile")
                
    return render_template('login.html')

@app.route('/voice_login', methods=['POST'])
def voice_login():
    """Handle voice-based login"""
    speak("Welcome to Voice Email System. Please say your username.")
    username = recognize_speech(is_username=True)  # Use special username mode
    
    if not username:
        return jsonify({'error': 'Could not recognize username'})
        
    # Check if user exists
    profile_path = Path("voice_auth") / f"{username}_profile.pkl"
    
    if profile_path.exists():
        # Existing user - verify voice
        speak(f"Welcome back {username}. Please verify your voice.")
        if authenticator.verify_user(username):
            speak("Voice verification successful!")
            session['username'] = username
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        else:
            speak("Voice verification failed. Please try again.")
            return jsonify({'error': 'Voice verification failed'})
    else:
        # New user - enroll voice
        speak(f"Welcome {username}. Let's create your voice profile.")
        if authenticator.enroll_user(username):
            speak("Voice profile created successfully!")
            session['username'] = username
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        else:
            speak("Failed to create voice profile. Please try again.")
            return jsonify({'error': 'Failed to create voice profile'})

if __name__ == '__main__':
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('voice_auth').mkdir(exist_ok=True)
    
    # Initialize logging
    logging.basicConfig(
        filename='logs/web_app.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n🌐 Server starting...")
    print("📱 Open your browser and go to: http://localhost:5001")
    print("🔊 Make sure your microphone is connected and permissions are granted\n")
    
    # Start the application
    try:
        socketio.run(app, debug=True, port=5001)
    finally:
        stop_gesture_recognition() 