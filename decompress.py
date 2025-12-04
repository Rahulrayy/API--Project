import soundfile as sf
import os

def flac_to_wav(input_flac):
    if not input_flac.lower().endswith(".flac"):
        raise ValueError("Input must be a .flac file")

    # Read FLAC
    data, samplerate = sf.read(input_flac)

    # Output filename
    output_wav = os.path.splitext(input_flac)[0] + ".wav"

    # Write WAV
    sf.write(output_wav, data, samplerate, format="WAV", subtype="PCM_16")

    return output_wav

# Example use
out = flac_to_wav("compressed.flac")
print("Saved:", out)
