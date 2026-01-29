import fnmatch
import io
import numpy as np
import os
import soundfile as sf
import torch
import warnings
import librosa

from torch.utils.data import Dataset


  

class WaveGlowDataset(Dataset):
    """WaveGlow dataset with support for metadata files."""

    def __init__(self,
                 audio_dir=None,
                 sample_rate=None,
                 local_condition_enabled=False,
                 local_condition_dir=None,
                 metadata_file=None):
        """Initializes the WaveGlowDataset.

        Args:
            audio_dir: Directory for audio data (used if metadata_file is None).
            sample_rate: Sample rate of audio.
            local_condition_enabled: Whether to use local condition.
            local_condition_dir: Directory for local condition (used if metadata_file is None).
            metadata_file: Path to metadata file with format:
                          path/to/mel.npy|text content
                          One entry per line.
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.local_condition_enabled = local_condition_enabled
        self.local_condition_dir = local_condition_dir
        self.metadata_file = metadata_file
        self.audio_files = []
        self.local_condition_files = []
        self.texts = []

        if metadata_file is not None:
            # Load from metadata file
            self._load_from_metadata(metadata_file)
        else:
            # Load from directories (original behavior)
            self.audio_files = sorted(self._find_files(audio_dir))
            if not self.audio_files:
                raise ValueError("No audio files found in '{}'.".format(audio_dir))

            if self.local_condition_enabled:
                self.local_condition_files = sorted(self._find_files(local_condition_dir, '*.npy'))
                self._check_consistency(self.audio_files, self.local_condition_files)

    def _load_from_metadata(self, metadata_file):
        """Load dataset from metadata file."""
        if not os.path.exists(metadata_file):
            raise ValueError("Metadata file not found: {}".format(metadata_file))
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Format: path/to/mel.npy|path/to/audio.wav
                parts = line.split('|')
                if len(parts) != 2:
                    raise ValueError(
                        "Invalid metadata format: {}. Expected: 'path/to/mel.npy|path/to/audio.wav'".format(line))
                
                mel_path, audio_path = parts
                mel_path = mel_path.strip()
                audio_path = audio_path.strip()
                
                # Resolve relative paths relative to metadata file directory
                metadata_dir = os.path.dirname(os.path.abspath(metadata_file))
                
                if not os.path.isabs(mel_path):
                    mel_path = os.path.join(metadata_dir, mel_path)
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join(metadata_dir, audio_path)
                
                if not os.path.exists(mel_path):
                    raise ValueError("Mel-spectrogram file not found: {}".format(mel_path))
                if not os.path.exists(audio_path):
                    raise ValueError("Audio file not found: {}".format(audio_path))
                
                self.audio_files.append(audio_path)
                self.local_condition_files.append(mel_path)

    def __len__(self):
        if self.metadata_file is not None:
            return len(self.local_condition_files)
        else:
            return len(self.audio_files)

    def __getitem__(self, index):
        if self.metadata_file is not None:
            # Load from metadata - both audio and mel-spec
            sample, sample_rate = sf.read(self.audio_files[index])
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                sample = self._resample_audio(sample, sample_rate, self.sample_rate)
            
            sample = sample.reshape(-1, 1)
            local_condition = np.load(self.local_condition_files[index])
            
            # Ensure mel-spectrogram is in correct format: (mel_bins, time_steps)
            # Assume files are saved as (time_steps, mel_bins) and always transpose
            if local_condition.ndim == 2 and local_condition.shape[1] == 80:
                # If second dimension is 80 (mel_bins), transpose
                local_condition = local_condition.T
            
            return sample, local_condition
        else:
            # Original behavior - load from audio files
            sample, sample_rate = sf.read(self.audio_files[index])
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                sample = self._resample_audio(sample, sample_rate, self.sample_rate)
            
            sample = sample.reshape(-1, 1)
            if self.local_condition_enabled:
                local_condition = np.load(self.local_condition_files[index])
                
                # Ensure mel-spectrogram is in correct format: (mel_bins, time_steps)
                # Assume files are saved as (time_steps, mel_bins) and always transpose
                if local_condition.ndim == 2 and local_condition.shape[1] == 80:
                    # If second dimension is 80 (mel_bins), transpose
                    local_condition = local_condition.T
                
                return sample, local_condition
            else:
                return sample
    
    def _resample_audio(self, audio, orig_sr, target_sr):
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


    def _find_files(self, directory, pattern='*.wav'):
        """Recursively finds all files matching the pattern."""
        files = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))
        return files

    def _check_consistency(self, audio_namelist, local_condition_namelist):
        audio_ids = [os.path.splitext(os.path.basename(name))[0]
                     for name in audio_namelist]
        local_condition_ids = [os.path.splitext(os.path.basename(name))[0]
                               for name in local_condition_namelist]
        if audio_ids != local_condition_ids:
            raise ValueError("Mismatch between audio files and local condition files.")


class WaveGlowCollate(object):
    """Function object used as a collate function for DataLoader."""

    def __init__(self, sample_size=None, upsample_factor=None, 
                 local_condition_enabled=None, use_metadata=False):
        """Initializes the WaveGlowCollate.

        Args:
            sample_size: The size of audio clips for training. 
            upsample_factor: The upsampling factor between sample and local condition.
            local_condition_enabled: Whether to use local condition.
            use_metadata: Whether dataset uses metadata files.
        """
        self.sample_size = sample_size
        self.upsample_factor = int(np.prod(upsample_factor)) if upsample_factor is not None else 1
        self.local_condition_enabled = local_condition_enabled
        self.use_metadata = use_metadata
        
        if sample_size is not None and upsample_factor is not None:
            if sample_size % self.upsample_factor != 0:
                raise ValueError(
                    "sample_size ({}) must be divisible by upsample_factor ({}).\n"
                    "Suggested values for upsample_factor={}: {}, {}, {}".format(
                        sample_size, self.upsample_factor, self.upsample_factor,
                        self.upsample_factor * (sample_size // self.upsample_factor),
                        self.upsample_factor * (sample_size // self.upsample_factor - 1),
                        self.upsample_factor * (sample_size // self.upsample_factor + 1)
                    )
                )

    def _collate_fn(self, batch):
        if self.use_metadata:
            # Batch format: (local_condition, text)
            return self._collate_metadata(batch)
        elif self.local_condition_enabled:
            # Original batch format: (sample, local_condition)
            return self._collate_audio(batch)
        else:
            # Just audio
            return self._collate_audio_only(batch)
    
    def _collate_metadata(self, batch):
        """Collate batch from metadata (audio + mel-specs)."""
        spec_frame_size = self.sample_size // self.upsample_factor
        
        new_batch = []
        for sample, local_condition in batch:
            # Ensure shapes are correct
            # sample: (time, 1)
            # local_condition: (mel_bins, time_steps)
            
            sample_len = len(sample)
            spec_len = local_condition.shape[1]
            
            # Align audio and spectrogram lengths
            # audio should be spec_frame_size * upsample_factor samples
            expected_audio_len = spec_frame_size * self.upsample_factor
            
            # Truncate mel-spectrogram if needed
            if spec_len > spec_frame_size:
                start = np.random.randint(0, spec_len - spec_frame_size)
                local_condition = local_condition[:, start:start + spec_frame_size]
                # Correspondingly crop audio
                audio_start = start * self.upsample_factor
                audio_end = audio_start + expected_audio_len
                sample = sample[audio_start:audio_end]
            elif spec_len < spec_frame_size:
                # Pad spectrogram
                pad_width = ((0, 0), (0, spec_frame_size - spec_len))
                local_condition = np.pad(local_condition, pad_width, mode='edge')
                # Pad audio
                pad_len = expected_audio_len - sample_len
                if pad_len > 0:
                    sample = np.pad(sample, ((0, pad_len), (0, 0)), mode='constant')
            else:
                # Ensure audio has correct length
                if sample_len > expected_audio_len:
                    sample = sample[:expected_audio_len]
                elif sample_len < expected_audio_len:
                    pad_len = expected_audio_len - sample_len
                    sample = np.pad(sample, ((0, pad_len), (0, 0)), mode='constant')
            
            new_batch.append((sample, local_condition))
        
        # Convert to proper format for training
        audio_batch = np.array([x[0] for x in new_batch])
        spec_batch = np.array([x[1] for x in new_batch])
        
        # Convert to tensors and transpose for model
        audio_batch = torch.FloatTensor(audio_batch).transpose(1, 2)  # (batch, 1, time)
        spec_batch = torch.FloatTensor(spec_batch)  # (batch, mel_bins, time)
        
        return audio_batch, spec_batch

    def _collate_audio(self, batch):
        """Collate batch from audio files (original behavior)."""
        new_batch = []
        for idx in range(len(batch)):
            sample, local_condition = batch[idx]

            # Pad utterance tail with silence, and cut tail if too much silence.
            sample_size = len(sample)
            local_condition_size = len(local_condition)
            length_diff = self.upsample_factor * local_condition_size - sample_size
            if length_diff > 0:
                sample = np.pad(
                    sample, [[0, length_diff], [0, 0]], 'constant')
            elif length_diff < 0:
                sample = sample[:self.upsample_factor * local_condition_size]

            if len(sample) > self.sample_size:
                frame_size  = self.sample_size // self.upsample_factor 
                lc_beg = np.random.randint(0, len(local_condition) - frame_size)
                sample_beg = lc_beg * self.upsample_factor
                sample = sample[sample_beg:sample_beg + self.sample_size, :]
                local_condition = local_condition[lc_beg:lc_beg + frame_size, :]
            new_batch.append((sample, local_condition))

        # Dynamic padding.
        max_len = max([len(x[0]) for x in new_batch])
        sample_batch = [
            np.pad(x[0], [[0, max_len - len(x[0])], [0, 0]], 'constant')
            for x in new_batch
        ]
        max_len = max([len(x[1]) for x in new_batch])
        local_condition_batch = [
            np.pad(x[1], [[0, max_len - len(x[1])], [0, 0]], 'edge')
            for x in new_batch
        ]

        # scalar output
        sample_batch = np.array(sample_batch)
        sample_batch = torch.FloatTensor(sample_batch).transpose(1, 2)

        # Local condition should be one timestep ahead of samples.
        local_condition_batch = torch.FloatTensor(
            np.array(local_condition_batch)).transpose(1, 2)

        return sample_batch, local_condition_batch
    
    def _collate_audio_only(self, batch):
        """Collate batch with only audio (no local conditions)."""
        new_batch = []
        for sample in batch:
            if len(sample) > self.sample_size:
                sample_beg = np.random.randint(0, len(sample) - self.sample_size)
                sample = sample[sample_beg : sample_beg + self.sample_size, :]
            new_batch.append(sample)

        # Dynamic padding.
        max_len = max([len(x) for x in new_batch])
        sample_batch = [
            np.pad(x, [[0, max_len - len(x)], [0, 0]], 'constant')
            for x in new_batch
        ]

        # scalar output
        sample_batch = np.array(sample_batch)
        sample_batch = torch.FloatTensor(sample_batch).transpose(1, 2)

        return sample_batch

    def __call__(self, batch):
        return self._collate_fn(batch)

