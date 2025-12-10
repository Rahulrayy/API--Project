# API-Project
Audio processing and indexing project Rahul Ray and Ivan Pavlovic

# Multi-Domain Audio Steganography

Hide secret text messages (up to 10 KB) inside WAV audio files. The audio sounds completely normal, but contains hidden encrypted data that can be extracted later.

## What It Does

This tool hides text inside audio files. To anyone listening, it sounds like normal audio. But if you know how to decode it, you can extract the hidden message. The system automatically verifies that messages can be successfully extracted and survives lossless compression like FLAC.

## Setup

Clone the repository and install dependencies:
```bash
git clone https://github.com/Rahulrayy/API--Project.git
cd API--Project
pip install -r requirements.txt
```

## Basic Usage

### Hide a Message

Run the encoder and follow the prompts:
```bash
python encoder.py
```

Press Enter twice to use defaults, then type your message. The encoder will create `stego.wav` with your hidden message inside.
Or use the given audio file ride.wav

### Extract a Message

Run the decoder:
```bash
python decoder.py
```

Press Enter to decode `stego.wav`. Your hidden message will be displayed and you can save it to a file.

### Test Compression

See if your message survives compression:
```bash
python compress.py      # Creates compressed.flac
python decompress.py    # Creates compressed.wav
python decoder.py       # Extract from compressed.wav
```

If you see "Hash Verified: True", the message survived compression perfectly.

## Files

**Main Scripts:**
- `encoder.py` - Hide messages in audio
- `decoder.py` - Extract hidden messages
- `compress.py` - Convert WAV to FLAC
- `decompress.py` - Convert FLAC back to WAV

## Limitations

- Maximum 10 KB message size
- Lossy compression (MP3/AAC) destroys the message
- XOR encryption is simple, not cryptographically secure
- Only works with WAV format audio

## Technical Details

Uses Discrete Wavelet Transform (DWT) to hide the message and Fast Fourier Transform (FFT) to store a SHA-256 verification hash. The encoder automatically tests that extraction works before saving the file.



---

Questions? Open an issue on [GitHub](https://github.com/Rahulrayy/API--Project).
