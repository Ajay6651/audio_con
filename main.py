import pyaudio
import wave
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def record_audio_and_save(file_path, duration=5):
    # Audio recording parameters
    FRAMES_PER_BUFFER = 3200
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open audio stream for recording
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    print("Start recording...")

    frames = []
    for _ in range(0, int(RATE / FRAMES_PER_BUFFER * duration)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    print("Recording stopped")

    # Stop the stream and close PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save recorded audio to a WAV file
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def perform_noise_cancellation(input_path, output_path):
    # Load the recorded audio
    audio = AudioSegment.from_wav(input_path)

    # Perform noise reduction
    reduced_audio = audio - 10  # Adjust the value for noise reduction

    # Print audio parameters
    print(f"Channels: {audio.channels}")
    print(f"Sample Width: {audio.sample_width}")
    print(f"Frame Rate: {audio.frame_rate}")
    print(f"Duration: {len(audio) / 1000.0} seconds")

    # Save the cleaned audio to a new file
    reduced_audio.export(output_path, format="wav")

    # Load the cleaned audio for feature extraction
    y, sr = librosa.load(output_path)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Print extracted features
    print("Extracted MFCCs:")
    print(mfccs)

    # Display the MFCCs
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mfccs, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency cepstral coefficients (MFCCs)')
    plt.show()

if __name__ == "__main__":
    # Specify the path for the recorded audio file
    recorded_audio_path = "recorded_audio.wav"

    # Specify the path for the cleaned audio output file
    cleaned_audio_path = "cleaned_audio.wav"

    # Record audio and save it to a file
    record_audio_and_save(recorded_audio_path)

    # Perform noise cancellation and save the cleaned audio
    perform_noise_cancellation(recorded_audio_path, cleaned_audio_path)

    print("Noise cancellation and feature extraction completed.")
