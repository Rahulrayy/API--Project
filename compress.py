import soundfile as sf

data, sr = sf.read("stego.wav")
sf.write("compressed.flac", data, sr, format="FLAC")
