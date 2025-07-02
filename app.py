import speech_recognition as sr
import traceback
import sys
import subprocess
import os
from pathlib import Path
import logging
import time
import multiprocessing as mp
import queue

from gmail_service import send_email_via_gmail, get_service
from semantic_search import run_semantic_search, get_most_recent_email
from gesture_process import gesture_recognition_process

class VoiceEmailSystem:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.max_retries = 3
        
        # Enhanced microphone settings for better accuracy
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 800  # Lower threshold for better sensitivity
        self.recognizer.pause_threshold = 0.8    # Longer pause threshold
        self.recognizer.phrase_threshold = 0.3   # More sensitive phrase detection
        self.recognizer.non_speaking_duration = 0.5  # Longer non-speaking duration
        self.speech_rate = 220
    
    def speak(self, text):
        """Speak the given text using macOS say command with faster rate"""
        try:
            print(f"[TTS] {text}")
            subprocess.run(['say', '-r', str(self.speech_rate), text], check=True)
            time.sleep(0.3)
        except Exception as e:
            print(f"Error in speak function: {e}")

    def recognize_speech(self, prompt=None, timeout=10, phrase_time_limit=5):
        """Enhanced speech recognition with improved accuracy and longer timeout"""
        for attempt in range(self.max_retries):
            try:
                with sr.Microphone() as source:
                    if prompt:
                        self.speak(prompt)
                        time.sleep(0.5)  # Give more time to prepare
                    
                    print("[System] Adjusting for ambient noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1.0)  # Longer adjustment
                    print("[System] Listening...")
                    
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                    print("[System] Processing speech...")
                    
                    text = self.recognizer.recognize_google(audio, language="en-US")
                    print(f"[You said] {text}")
                    
                    # Clean up the recognized text
                    text = text.lower().strip()
                    
                    # Handle partial commands with more lenient matching
                    if len(text) >= 2:  # Only process if we have at least 2 characters
                        if text.startswith("comp") or text == "compose":
                            text = "compose"
                        elif text.startswith("sear") or text == "search":
                            text = "search"
                        elif text.startswith("rec") or text == "recent":
                            text = "recent"
                        elif text.startswith("ex") or text == "exit":
                            text = "exit"
                        elif text.startswith("conf") or text == "confirm":
                            text = "confirm"
                        elif text.startswith("can") or text == "cancel":
                            text = "cancel"
                    
                    return text
                    
            except sr.WaitTimeoutError:
                print("[System] Listening timed out. Please try again.")
            except sr.UnknownValueError:
                print("[System] Could not understand audio")
            except sr.RequestError as e:
                print(f"[System] Could not request results; {e}")
            except Exception as e:
                print(f"[System] Error in speech recognition: {e}")
            
            if attempt < self.max_retries - 1:
                self.speak("I didn't catch that. Please try again.")
            
        return None

    def get_email_details(self):
        """Get email details with validation"""
        # Get recipient email with longer timeout
        for _ in range(self.max_retries):
            self.speak("Please prepare to say the recipient's email address.")
            time.sleep(1)  # Give user time to prepare
            self.speak("Listening now...")
            
            email = self.recognize_speech(timeout=10, phrase_time_limit=7)
            if not email:
                self.speak("I didn't hear an email address. Let's try again.")
                continue
                
            email = email.replace(" at ", "@").replace(" dot ", ".").replace(" ", "")
            self.speak(f"I heard the email address as: {email}")
            self.speak("To confirm this email address, say 'confirm'. To try again, say 'cancel'.")
            
            with sr.Microphone() as source:
                try:
                    print("[System] Listening for confirmation...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=2)
                    confirmation = self.recognizer.recognize_google(audio, language="en-US").lower()
                    print(f"[Confirmation] {confirmation}")
                    
                    if 'confirm' in confirmation:
                        break
                    elif 'cancel' in confirmation:
                        if _ < self.max_retries - 1:
                            self.speak("Let's try again.")
                        continue
                    else:
                        self.speak("Please say either confirm or cancel.")
                        continue
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    self.speak("I didn't hear your response. Please say confirm or cancel clearly.")
                    continue
                except Exception as e:
                    print(f"Error in confirmation: {e}")
                    continue
            
        else:  # If we've exhausted all retries
            self.speak("Too many unsuccessful attempts. Cancelling email composition.")
            return None, None, None

        # Get subject
        self.speak("Please dictate the subject of your email.")
        subject = self.recognize_speech(timeout=7)
        if not subject:
            return None, None, None

        # Get body
        self.speak("Now please dictate the content of your email.")
        body = self.recognize_speech(phrase_time_limit=30)
        if not body:
            return None, None, None

        return email, subject, body

    def handle_email_search(self):
        """Enhanced email search functionality"""
        self.speak("What kind of emails would you like to search for?")
        query = self.recognize_speech()
        
        if not query:
            self.speak("No search query provided.")
            return False
        
        try:
            matching_emails = run_semantic_search(query)
            
            if matching_emails:
                self.speak(f"I found {len(matching_emails)} matching emails. I'll read them to you now.")
                
                for i, match in enumerate(matching_emails, 1):
                    email = match['email']
                    similarity = match['similarity']
                    
                    self.speak(f"Email {i} with {int(similarity * 100)}% relevance:")
                    self.speak(f"From: {email['from']}")
                    self.speak(f"Subject: {email['subject']}")
                    self.speak("Message content:")
                    self.speak(email['body'])
                    
                    if i < len(matching_emails):
                        self.speak("Would you like to hear the next email? Say yes or no.")
                        response = self.recognize_speech(timeout=3)
                        if not response or 'no' in response:
                            break
            else:
                self.speak("No matching emails found.")
            return True
            
        except Exception as e:
            print(f"Error in email search: {e}")
            self.speak("Sorry, I encountered an error while searching emails.")
            return False

    def handle_most_recent_email(self):
        """Get most recent email"""
        self.speak("Fetching your most recent email...")
        try:
            result = get_most_recent_email()
            
            if result:
                formatted_email, email_content = result
                self.speak("Here is your most recent email:")
                self.speak(f"From: {email_content['from']}")
                self.speak(f"Subject: {email_content['subject']}")
                self.speak("Message content:")
                self.speak(email_content['body'])
                return True
            else:
                self.speak("Sorry, I couldn't fetch your most recent email.")
                return False
        except Exception as e:
            print(f"Error fetching recent email: {e}")
            self.speak("Sorry, I encountered an error while fetching your recent email.")
            return False

    def handle_email_composition(self, username):
        """Handle email composition without voice authentication"""
        self.speak("Let's compose your email.")
        email_details = self.get_email_details()
        
        if not email_details:
            self.speak("Email composition cancelled.")
            return False
            
        email, subject, body = email_details
        
        # Final confirmation with gesture
        self.speak("Here's your email. Please review:")
        self.speak(f"To: {email}")
        self.speak(f"Subject: {subject}")
        self.speak(f"Content: {body}")
        self.speak("Please hold your hand up clearly in front of the camera.")
        time.sleep(1)
        self.speak("You can either show thumbs up to send, palm to cancel, or say 'confirm' to send, 'cancel' to cancel.")

        # Initialize gesture recognition
        gesture_queue = mp.Queue()
        stop_event = mp.Event()
        gesture_process = mp.Process(
            target=gesture_recognition_process,
            args=(gesture_queue, stop_event)
        )

        email_sent = False
        try:
            # Start gesture recognition
            gesture_process.start()
            print("\n📸 Starting camera and gesture recognition...")
            time.sleep(2)  # Give camera time to initialize
            
            print("\n👋 Ready for gestures and voice commands!")
            print("Options:")
            print("1. 👍 Thumbs up  -> send email")
            print("2. ✋ Palm       -> cancel email")
            print("3. 🗣️  Say 'confirm' -> send email")
            print("4. 🗣️  Say 'cancel'  -> cancel email")
            print("\n💡 Tips:")
            print("- Hold your hand steady and clearly visible")
            print("- Keep your hand about 1-2 feet from the camera")
            print("- Ensure good lighting")
            print("- Or use voice commands clearly\n")

            start_time = time.time()
            timeout = 30  # Extended timeout for better detection
            last_feedback_time = time.time()
            feedback_interval = 5  # Give feedback every 5 seconds

            while time.time() - start_time < timeout:
                # Periodic feedback
                current_time = time.time()
                if current_time - last_feedback_time >= feedback_interval:
                    print("👀 Watching for gestures and listening for commands...")
                    last_feedback_time = current_time

                # Check for gesture
                try:
                    gesture = gesture_queue.get_nowait()
                    print(f"\n✨ Gesture detected: {gesture}")
                    if gesture == "send":
                        print("👍 Thumbs up recognized!")
                        self.speak("Thumbs up detected. Sending email now...")
                        try:
                            send_email_via_gmail(email, subject, body)
                            self.speak("Email sent successfully!")
                            email_sent = True
                            break
                        except Exception as e:
                            print(f"\n❌ Error sending email: {e}")
                            self.speak("Sorry, I encountered an error while sending your email.")
                            break
                    elif gesture == "cancel":
                        print("✋ Palm recognized!")
                        self.speak("Palm detected. Cancelling email.")
                        break
                except queue.Empty:
                    pass

                # Check for voice command
                try:
                    with sr.Microphone() as source:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=2)
                        try:
                            response = self.recognizer.recognize_google(audio).lower()
                            print(f"\n🗣️ Voice command detected: {response}")
                            
                            if 'confirm' in response:
                                print("Voice confirmation received!")
                                self.speak("Voice confirmation received. Sending email now...")
                                try:
                                    send_email_via_gmail(email, subject, body)
                                    self.speak("Email sent successfully!")
                                    email_sent = True
                                    break
                                except Exception as e:
                                    print(f"\n❌ Error sending email: {e}")
                                    self.speak("Sorry, I encountered an error while sending your email.")
                                    break
                            elif 'cancel' in response:
                                print("Voice cancellation received!")
                                self.speak("Cancelling email as requested.")
                                break
                        except sr.UnknownValueError:
                            pass  # No speech detected
                except sr.WaitTimeoutError:
                    pass  # Timeout on listen
                except Exception as e:
                    print(f"Error in voice recognition: {e}")
                    pass

                time.sleep(0.1)  # Prevent CPU overuse

            if not email_sent and time.time() - start_time >= timeout:
                print("\n⏰ Timeout reached")
                self.speak("No confirmation received within time limit. Cancelling email.")

        except Exception as e:
            print(f"\n❌ Error during recognition: {e}")
            self.speak("An error occurred. Cancelling email.")

        finally:
            # Cleanup gesture recognition
            if not stop_event.is_set():
                stop_event.set()
            if gesture_process.is_alive():
                gesture_process.join(timeout=2)
                if gesture_process.is_alive():
                    gesture_process.terminate()
                gesture_process.join()
            print("\n🎥 Gesture recognition cleaned up")

        return email_sent

    def get_username(self):
        """Get username from voice input"""
        self.speak("Please say your username.")
        username = self.recognize_speech(timeout=5)
        if username:
            # Clean up username: remove spaces and convert to lowercase
            username = username.lower().replace(" ", "")
            self.speak(f"I heard the username: {username}")
            self.speak("To confirm this username, say 'proceed'. To try again, say 'retry'.")
            
            confirmation = self.recognize_speech(timeout=5)
            if confirmation and 'proceed' in confirmation:
                return username
        return None

    def handle_voice_enrollment(self, username):
        """Handle voice profile enrollment"""
        self.speak(f"Hello {username}. Let's create your voice profile.")
        
        from voice_auth import VoiceAuthenticator
        authenticator = VoiceAuthenticator()
        
        if authenticator.enroll_user(username):
            self.speak("Voice profile created successfully!")
            return True
        else:
            self.speak("There was a problem creating your voice profile.")
            self.speak("Would you like to try again? Say yes or no.")
            retry = self.recognize_speech(timeout=5)
            if retry and 'yes' in retry.lower():
                return self.handle_voice_enrollment(username)
            return False

    def authenticate_user(self, username):
        """Authenticate user with voice"""
        from voice_auth import VoiceAuthenticator
        authenticator = VoiceAuthenticator()
        
        # Check if profile exists
        profile_path = authenticator.voice_dir / f"{username}_profile.pkl"
        if not profile_path.exists():
            self.speak(f"No voice profile found for username: {username}")
            self.speak("Would you like to create a new profile? Say 'create' or 'skip'.")
            response = self.recognize_speech(timeout=5)
            if response and 'create' in response:
                return self.handle_voice_enrollment(username)
            return False
            
        # Verify voice
        for attempt in range(3):
            self.speak(f"Please verify your voice for username: {username}")
            if authenticator.verify_user(username):
                self.speak("Voice authentication successful!")
                return True
            else:
                if attempt < 2:
                    self.speak("Voice authentication failed. Say 'continue' to try again or 'stop' to cancel.")
                    response = self.recognize_speech(timeout=5)
                    if not response or 'stop' in response:
                        return False
                else:
                    self.speak("Voice authentication failed too many times.")
        return False

    def run(self):
        """Main application loop"""
        print("\n🚀 Starting Voice-Based Email System...")
        self.speak("Welcome to Voice Email System!")
        
        while True:
            try:
                # Get username
                username = self.get_username()
                if not username:
                    continue
                
                # Check if user exists
                profile_path = Path("voice_auth") / f"{username}_profile.pkl"
                
                if not profile_path.exists():
                    # New user - enroll voice
                    self.speak(f"Welcome {username}! Let's create your voice profile.")
                    if not self.handle_voice_enrollment(username):
                        continue
                else:
                    # Existing user - verify voice
                    self.speak(f"Welcome back {username}! Let's verify your voice.")
                    if not self.authenticate_user(username):
                        continue
                
                # Main menu loop
                while True:
                    self.speak("What would you like to do? Say compose, search, recent, or exit.")
                    command = self.recognize_speech(timeout=5)
                    
                    if not command:
                        self.speak("I didn't hear a command. Please try again.")
                        continue
                    
                    if command == "compose":
                        self.handle_email_composition(username)
                    elif command == "search":
                        self.handle_email_search()
                    elif command == "recent":
                        self.handle_most_recent_email()
                    elif command == "exit":
                        self.speak("Goodbye!")
                        return
                    else:
                        self.speak("I didn't understand that command. Please try again.")
                
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                traceback.print_exc()
                self.speak("An error occurred. Let's try again.")

def main():
    try:
        # Needed for macOS
        mp.set_start_method('spawn', force=True)
        
        # Create necessary directories
        Path('logs').mkdir(exist_ok=True)
        Path('voice_auth').mkdir(exist_ok=True)
            
        # Check for credentials
        if not Path('credentials.json').exists():
            print("\n[ERROR] credentials.json not found!")
            print("Please follow these steps:")
            print("1. Go to Google Cloud Console")
            print("2. Create a project and enable Gmail API")
            print("3. Create OAuth 2.0 credentials")
            print("4. Download and save as 'credentials.json' in the project root")
            return
            
        # Initialize logging
        logging.basicConfig(
            filename='logs/app.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Check audio device
        try:
            with sr.Microphone() as source:
                print("\n✅ Audio device initialized successfully")
        except Exception as e:
            print(f"\n[ERROR] Audio device initialization failed: {e}")
            print("Please check your microphone settings and try again.")
            return
        
        # Initialize and run the system
        system = VoiceEmailSystem()
        system.run()
        
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        if 'logging' in locals():
            logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
