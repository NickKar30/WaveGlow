import os
import argparse
import csv
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm


def prepare_mel(audio_path, sr=22050, n_fft=1024, n_mels=80, hop_length=256, 
                fmin=0, fmax=11025, normalized=True, ref_power=20.0):
    """
    Convert audio to mel-spectrogram.
    
    Args:
        audio_path (str): Path to audio file
        sr (int): Sample rate
        n_fft (int): FFT window size
        n_mels (int): Number of mel bands (should be 80 for Tacotron2)
        hop_length (int): Number of samples between successive frames
        fmin (int): Minimum frequency
        fmax (int): Maximum frequency
        normalized (bool): Whether to normalize using dB scale
        ref_power (float): Reference power for dB normalization
    
    Returns:
        mel (np.ndarray): Mel-spectrogram of shape (T, n_mels)
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        mel = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )
        
        if normalized:
            mel = librosa.power_to_db(mel, ref=np.max)
            mel = (mel - (-80.0)) / (0.0 - (-80.0))
            mel = np.clip(mel, 0, 1)
        
        mel = mel.T.astype(np.float32)
        
        return mel, sr
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None


def process_dataset(audio_dir, output_dir, metadata_file, output_csv, sr=22050):
    """
    Process audio files and create mel-spectrograms.
    
    Expected metadata file format:
        audio_file.wav|Text transcription in Russian
        
    Args:
        audio_dir (str): Directory containing audio files
        output_dir (str): Directory to save mel-spectrograms
        metadata_file (str): Path to metadata file with audio and text
        output_csv (str): Path to output CSV file for training
        sr (int): Sample rate
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    audio_path = Path(audio_dir)
    output_path = Path(output_dir)
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        data = list(reader)
    
    csv_lines = []
    
    print(f"Processing {len(data)} audio files...")
    
    for i, row in enumerate(tqdm(data)):
        
        if len(row) < 2:
            print(f"Skipping malformed line {i}: {row}")
        
        file_id = row[0]
        text = row[1]
        
        if not file_id.endswith('.wav'):
            audio_file = f"{file_id}.wav"
        else:
            audio_file = file_id
            
        full_audio_path = audio_path / audio_file
        
        if not full_audio_path.exists():
            print(f"Audio file not found: {full_audio_path}")
            continue
        
        mel, _ = prepare_mel(str(full_audio_path), sr=sr)
        
        if mel is None:
            print(f"Failed to process {full_audio_path}")
            continue
        
        mel_filename = f"{Path(audio_file).stem}.npy"
        mel_save_path = output_path / mel_filename
        np.save(str(mel_save_path), mel)
        
        csv_lines.append(f"{mel_filename}|{text}")
    
    if os.path.dirname(output_csv):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write('\n'.join(csv_lines))
    
    print(f"Processed {len(csv_lines)} files")
    print(f"Mel-spectrograms saved to: {output_dir}")
    print(f"CSV file saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare mel-spectrograms for Tacotron2 training'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        default='/home/jovyan/shares/SR008.fs2/nkaragodin/DATASET/audio22k',
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='DATASET/mels_22k',
        help='Directory to save mel-spectrograms'
    )
    parser.add_argument(
        '--metadata_file',
        type=str,
        help='Path to metadata file (format: audio_file|text)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='DATASET/train.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=22050,
        help='Sample rate (default: 22050 Hz)'
    )
    
    args = parser.parse_args()
    
    process_dataset(
        args.audio_dir,
        args.output_dir,
        args.metadata_file,
        args.output_csv,
        sr=args.sr
    )


if __name__ == '__main__':
    main()
