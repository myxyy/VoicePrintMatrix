import librosa
import matplotlib.pyplot as plt
import numpy as np

# generate spectrogram and save as a png file
def generate_spectrogram(audio_path, output_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Generate the spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=22050)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    # Example usage
    audio_file = 'resources/zundamon_reconstructed.wav'
    output_image = 'resources/spec_zunda_reconstructed.png'
    generate_spectrogram(audio_file, output_image)
    print(f"Spectrogram saved to {output_image}")