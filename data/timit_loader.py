"""TIMIT database loader for speech dereverberation experiments.

The TIMIT Acoustic-Phonetic Continuous Speech Corpus is a widely used dataset
for speech recognition research. This module provides utilities to access TIMIT
data for dereverberation experiments.

Note: TIMIT is licensed by the Linguistic Data Consortium (LDC). You must have
a valid license to use this dataset. Alternative free datasets are also supported.
"""

import os
import warnings
from typing import Tuple, Optional, List
import numpy as np

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    warnings.warn("torchaudio not available. Install with: pip install torchaudio")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class TIMITLoader:
    """
    Loader for TIMIT dataset with fallback to alternative free speech datasets.
    
    TIMIT License Information:
    - TIMIT is distributed by the Linguistic Data Consortium (LDC)
    - Catalog Number: LDC93S1
    - License required: https://catalog.ldc.upenn.edu/LDC93S1
    - Price: ~$250 for academic users
    
    Free Alternatives:
    - LibriSpeech (http://www.openslr.org/12/)
    - Common Voice (https://commonvoice.mozilla.org/)
    - VCTK (https://datashare.ed.ac.uk/handle/10283/3443)
    """
    
    def __init__(
        self,
        root_dir: Optional[str] = None,
        use_alternative: bool = True,
        sample_rate: int = 16000
    ):
        """
        Initialize TIMIT loader.
        
        Args:
            root_dir: Path to TIMIT dataset root (if available)
            use_alternative: If True and TIMIT not found, use free alternative
            sample_rate: Target sample rate for audio
        """
        self.root_dir = root_dir
        self.use_alternative = use_alternative
        self.sample_rate = sample_rate
        self.timit_available = False
        
        # Check if TIMIT is available
        if root_dir and os.path.exists(root_dir):
            self.timit_available = True
            print(f"TIMIT dataset found at {root_dir}")
        else:
            print("TIMIT dataset not found.")
            if use_alternative:
                print("Will use alternative free speech dataset (LibriSpeech).")
                self._setup_alternative()
    
    def _setup_alternative(self):
        """Setup alternative free speech dataset."""
        if not TORCHAUDIO_AVAILABLE:
            warnings.warn(
                "torchaudio is required for automatic dataset download. "
                "Install with: pip install torchaudio"
            )
            return
        
        # Use LibriSpeech as alternative
        self.alternative_name = "LibriSpeech"
        print("Alternative dataset: LibriSpeech (free, open license)")
        print("License: CC BY 4.0")
        print("Citation: Panayotov et al. (2015)")
    
    def load_utterance(
        self,
        utterance_path: str
    ) -> Tuple[np.ndarray, int]:
        """
        Load a speech utterance from file.
        
        Args:
            utterance_path: Path to audio file
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        if LIBROSA_AVAILABLE:
            waveform, sr = librosa.load(utterance_path, sr=self.sample_rate)
            return waveform, sr
        elif TORCHAUDIO_AVAILABLE:
            waveform, sr = torchaudio.load(utterance_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            return waveform.numpy().squeeze(), self.sample_rate
        else:
            raise ImportError(
                "Either librosa or torchaudio is required. "
                "Install with: pip install librosa OR pip install torchaudio"
            )
    
    def get_librispeech(
        self,
        subset: str = "test-clean",
        download: bool = True,
        download_dir: str = "./data/LibriSpeech"
    ):
        """
        Get LibriSpeech dataset as free alternative to TIMIT.
        
        Args:
            subset: LibriSpeech subset ('test-clean', 'dev-clean', 'train-clean-100', etc.)
            download: Whether to download if not present
            download_dir: Directory to download dataset to
            
        Returns:
            LibriSpeech dataset object (torchaudio.datasets.LIBRISPEECH)
        """
        if not TORCHAUDIO_AVAILABLE:
            raise ImportError("torchaudio required for LibriSpeech. Install: pip install torchaudio")
        
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=download_dir,
            url=subset,
            download=download
        )
        
        print(f"LibriSpeech {subset} loaded with {len(dataset)} utterances")
        print("License: CC BY 4.0 (free to use)")
        
        return dataset
    
    def get_test_utterances(
        self,
        num_utterances: int = 100,
        max_duration: float = 5.0
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Get test utterances for dereverberation experiments.
        
        Args:
            num_utterances: Number of utterances to retrieve
            max_duration: Maximum duration in seconds
            
        Returns:
            List of (waveform, sample_rate) tuples
        """
        utterances = []
        
        if self.timit_available:
            # Load from TIMIT
            utterances = self._load_timit_utterances(num_utterances, max_duration)
        elif self.use_alternative:
            # Load from LibriSpeech
            utterances = self._load_librispeech_utterances(num_utterances, max_duration)
        else:
            warnings.warn("No speech dataset available")
        
        return utterances
    
    def _load_timit_utterances(
        self,
        num_utterances: int,
        max_duration: float
    ) -> List[Tuple[np.ndarray, int]]:
        """Load utterances from TIMIT dataset."""
        utterances = []
        
        # TIMIT structure: TIMIT/TEST/DR1/FCJF0/SA1.WAV
        test_dir = os.path.join(self.root_dir, "TEST")
        
        if not os.path.exists(test_dir):
            warnings.warn(f"TIMIT test directory not found: {test_dir}")
            return utterances
        
        count = 0
        for dialect_region in os.listdir(test_dir):
            dr_path = os.path.join(test_dir, dialect_region)
            if not os.path.isdir(dr_path):
                continue
            
            for speaker_id in os.listdir(dr_path):
                speaker_path = os.path.join(dr_path, speaker_id)
                if not os.path.isdir(speaker_path):
                    continue
                
                for wav_file in os.listdir(speaker_path):
                    if not wav_file.endswith('.WAV'):
                        continue
                    
                    wav_path = os.path.join(speaker_path, wav_file)
                    waveform, sr = self.load_utterance(wav_path)
                    
                    # Filter by duration
                    duration = len(waveform) / sr
                    if duration <= max_duration:
                        utterances.append((waveform, sr))
                        count += 1
                    
                    if count >= num_utterances:
                        return utterances
        
        return utterances
    
    def _load_librispeech_utterances(
        self,
        num_utterances: int,
        max_duration: float
    ) -> List[Tuple[np.ndarray, int]]:
        """Load utterances from LibriSpeech dataset."""
        if not TORCHAUDIO_AVAILABLE:
            warnings.warn("torchaudio required for LibriSpeech")
            return []
        
        dataset = self.get_librispeech(subset="test-clean", download=True)
        utterances = []
        
        for i, (waveform, sr, transcript, speaker_id, chapter_id, utterance_id) in enumerate(dataset):
            if i >= num_utterances:
                break
            
            # Convert to numpy and check duration
            waveform_np = waveform.numpy().squeeze()
            duration = len(waveform_np) / sr
            
            if duration <= max_duration:
                # Resample if needed
                if sr != self.sample_rate:
                    if LIBROSA_AVAILABLE:
                        waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=self.sample_rate)
                    else:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform_np = resampler(waveform).numpy().squeeze()
                
                utterances.append((waveform_np, self.sample_rate))
        
        return utterances


def check_timit_license():
    """
    Print information about TIMIT licensing.
    """
    print("=" * 70)
    print("TIMIT Acoustic-Phonetic Continuous Speech Corpus")
    print("=" * 70)
    print("\nLicense Information:")
    print("- Distributor: Linguistic Data Consortium (LDC)")
    print("- Catalog Number: LDC93S1")
    print("- License Type: Commercial (requires purchase)")
    print("- Price: ~$250 USD for academic users, ~$2500 for commercial")
    print("- Website: https://catalog.ldc.upenn.edu/LDC93S1")
    print("\nUsage:")
    print("- You must purchase a license from LDC to use TIMIT")
    print("- Academic users can apply for membership discounts")
    print("- Dataset contains 630 speakers, 8 dialects, ~5 hours of speech")
    print("\nFree Alternatives:")
    print("1. LibriSpeech (http://www.openslr.org/12/)")
    print("   - 1000 hours of read English speech")
    print("   - License: CC BY 4.0 (free)")
    print("   - Can be downloaded automatically via torchaudio")
    print("\n2. Common Voice (https://commonvoice.mozilla.org/)")
    print("   - Multiple languages")
    print("   - License: CC0 (public domain)")
    print("\n3. VCTK (https://datashare.ed.ac.uk/handle/10283/3443)")
    print("   - 110 speakers, British accents")
    print("   - License: CC BY 4.0 (free)")
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    # Print license information
    check_timit_license()
    
    print("\n" + "=" * 70)
    print("Recommended approach for this project:")
    print("=" * 70)
    print("\nUse LibriSpeech as a free alternative to TIMIT:")
    print("```python")
    print("loader = TIMITLoader(use_alternative=True)")
    print("utterances = loader.get_test_utterances(num_utterances=100)")
    print("```")
    print("\nThis will automatically download LibriSpeech test-clean subset")
    print("and provide clean speech utterances for dereverberation experiments.")
