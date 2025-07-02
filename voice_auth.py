import os
import numpy as np
import pyaudio
import pickle
from pathlib import Path
import time
import warnings
import traceback
import subprocess
import torch
import torchaudio
from speechbrain.inference import EncoderClassifier
from scipy.spatial.distance import cosine
warnings.filterwarnings('ignore')

class VoiceAuthenticator:
    def __init__(self):
        self.sample_rate = 16000
        self.duration = 4  # Adjusted duration
        self.voice_dir = Path("voice_auth")
        self.voice_dir.mkdir(exist_ok=True)
        self.threshold = 0.55  # Lowered threshold for better acceptance
        self.max_attempts = 3  # Allow up to 3 attempts
        
        # Initialize the pre-trained speaker recognition model
        print("\n📥 Loading pre-trained speaker recognition model...")
        self.speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        print("✅ Model loaded successfully!")
        
        # Simpler verification phrases
        self.verification_phrases = [
            "Hello this is my voice",
            "I am using my email",
            "Open my email account",
            "Access my inbox now",
            "Check my messages please"
        ]
        
        self.current_phrase = None
        
    def record_audio(self, duration=None):
        """Record audio from microphone using PyAudio"""
        if duration is None:
            duration = self.duration
        
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=self.sample_rate,
                          input=True,
                          frames_per_buffer=CHUNK)
            
            print("🎤 Recording started...")
            frames = []
            
            # Record audio immediately
            for i in range(0, int(self.sample_rate / CHUNK * duration)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            print("✅ Recording completed")
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Convert to numpy array
            audio_data = b''.join(frames)
            audio = np.frombuffer(audio_data, dtype=np.float32)
            
            # Normalize audio
            if len(audio) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio
            
        except Exception as e:
            print(f"❌ Error recording audio: {e}")
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            if 'p' in locals():
                p.terminate()
            return None

    def extract_embeddings(self, audio):
        """Extract speaker embeddings using the pre-trained model"""
        try:
            if audio is None or len(audio) == 0:
                print("❌ No audio data to process")
                return None
                
            # Convert numpy array to torch tensor
            waveform = torch.FloatTensor(audio).unsqueeze(0)
            
            # Extract embeddings
            embeddings = self.speaker_model.encode_batch(waveform)
            
            # Convert to numpy and flatten
            embeddings_np = embeddings.squeeze().cpu().numpy()
            
            return embeddings_np
            
        except Exception as e:
            print(f"❌ Error extracting embeddings: {e}")
            traceback.print_exc()
            return None
            
    def enroll_user(self, username, audio=None, num_samples=5):
        """Enroll a new user with multiple voice samples using different phrases"""
        # Validate username
        if not username or len(username) < 2 or len(username) > 20:
            print("❌ Invalid username length")
            return False
            
        # Check for system messages
        invalid_phrases = [
            "welcome to voice email system",
            "voice email system",
            "welcome",
            "system",
            "email",
            "voice"
        ]
        
        if any(phrase in username.lower() for phrase in invalid_phrases):
            print("❌ Invalid username: contains system message")
            return False
            
        print(f"\n📝 Creating voice profile for: {username}")
        print("\n💡 Tips for clear recording:")
        print("- Speak naturally at a normal pace")
        print("- Keep consistent distance from microphone")
        print("- Minimize background noise")
        
        embeddings_list = []
        phrases_used = []
        
        for i in range(num_samples):
            # Select a random phrase that hasn't been used yet
            available_phrases = [p for p in self.verification_phrases if p not in phrases_used]
            if not available_phrases:
                phrases_used = []  # Reset if we've used all phrases
                available_phrases = self.verification_phrases
                
            phrase = np.random.choice(available_phrases)
            phrases_used.append(phrase)
            
            print(f"\n🎤 Recording {i+1} of {num_samples}")
            self.speak(f"Please say: {phrase}")
            time.sleep(1)  # Give user time to prepare
            
            # If audio is provided (from web_app), use it directly
            if audio is not None and i == 0:
                current_audio = audio
            else:
                current_audio = self.record_audio()
                
            if current_audio is not None:
                embeddings = self.extract_embeddings(current_audio)
                if embeddings is not None:
                    embeddings_list.append(embeddings)
                    print(f"✅ Recording {i+1} successful")
                    continue
            
            print(f"❌ Recording failed. Let's try again.")
            i -= 1  # Retry this sample
            phrases_used.pop()  # Remove the failed phrase
                
        if len(embeddings_list) == num_samples:  # Only proceed if we have all samples
            try:
                # Stack all embeddings
                embeddings_array = np.vstack(embeddings_list)
                
                # Calculate mean embedding (voice profile)
                mean_embedding = np.mean(embeddings_array, axis=0)
                
                # Normalize the mean embedding
                mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
                
                # Debug: Print embedding statistics
                print("\n📊 Debug: Enrollment Statistics")
                print(f"Number of voice samples: {len(embeddings_list)}")
                print(f"Embedding dimension: {mean_embedding.shape}")
                
                # Calculate voice profile
                voice_profile = {
                    'mean_embedding': mean_embedding,
                    'phrases': self.verification_phrases.copy(),
                    'used_phrases': []  # Track used phrases for verification
                }
                
                # Save voice profile
                profile_path = self.voice_dir / f"{username}_profile.pkl"
                with open(profile_path, 'wb') as f:
                    pickle.dump(voice_profile, f)
                    
                print(f"\n✅ Voice profile created successfully for {username}")
                self.speak("Voice profile created successfully!")
                return True
                
            except Exception as e:
                print(f"\n❌ Error creating voice profile: {e}")
                traceback.print_exc()
                return False
        
        print("\n❌ Failed to create voice profile")
        return False
        
    def verify_user(self, username):
        """Verify a user's voice with enhanced security using random phrases"""
        profile_path = self.voice_dir / f"{username}_profile.pkl"
        
        if not profile_path.exists():
            print(f"❌ No voice profile found for {username}")
            return False
            
        # Load stored profile
        try:
            with open(profile_path, 'rb') as f:
                voice_profile = pickle.load(f)
                
            stored_embedding = voice_profile['mean_embedding']
            print(f"\n📊 Stored embedding dimension: {stored_embedding.shape}")
            
            # Get list of phrases that haven't been used recently
            used_phrases = voice_profile.get('used_phrases', [])
            available_phrases = [p for p in self.verification_phrases if p not in used_phrases[-2:]]
            
            if not available_phrases:
                available_phrases = self.verification_phrases
                
            # Select a random phrase
            self.current_phrase = np.random.choice(available_phrases)
            
            # Update used phrases
            voice_profile['used_phrases'] = used_phrases[-2:] + [self.current_phrase]
            
            # Save updated profile
            with open(profile_path, 'wb') as f:
                pickle.dump(voice_profile, f)
                
        except Exception as e:
            print(f"Error loading profile: {e}")
            traceback.print_exc()
            return False
            
        print("\n🔐 Voice Verification")
        print(f"Please say: \"{self.current_phrase}\"")
        self.speak(f"Please say: {self.current_phrase}")
        
        # Give user time to prepare
        time.sleep(1)
        
        # Record and verify
        audio = self.record_audio()
        if audio is None:
            return False
            
        # Extract embeddings
        embeddings = self.extract_embeddings(audio)
        if embeddings is None:
            return False
        
        # Calculate similarity
        similarity = 1 - cosine(stored_embedding, embeddings)
        print(f"\n📊 Similarity score: {similarity:.3f}")
        print(f"📊 Threshold: {self.threshold}")
        
        # More lenient verification with multiple attempts
        if similarity >= self.threshold:
            print("✅ Voice verification successful!")
            return True
        elif similarity >= 0.45:  # Additional check for borderline cases
            print("⚠️ Borderline match detected. Please try again.")
            return False
        else:
            print("❌ Voice verification failed")
            return False
        
    def delete_profile(self, username):
        """Delete a user's voice profile"""
        profile_path = self.voice_dir / f"{username}_profile.pkl"
        
        if profile_path.exists():
            profile_path.unlink()
            print(f"\n🗑️ Voice profile deleted for {username}")
            return True
            
        print(f"❌ No voice profile found for {username}")
        return False
        
    def delete_all_profiles(self):
        """Delete all voice profiles from the voice_auth directory"""
        try:
            # Delete all .pkl files in the voice_auth directory
            for profile in self.voice_dir.glob("*.pkl"):
                profile.unlink()
            print("\n🗑️ All voice profiles have been deleted")
            return True
        except Exception as e:
            print(f"\n❌ Error deleting profiles: {e}")
            return False
        
    def speak(self, text):
        """Speak the given text using macOS say command"""
        try:
            print(f"[TTS] {text}")
            subprocess.run(['say', '-r', '220', text], check=True)
            time.sleep(0.3)
        except Exception as e:
            print(f"❌ Error in speak function: {e}") 