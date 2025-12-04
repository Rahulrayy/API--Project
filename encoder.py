import numpy as np
import pywt
import hashlib
import soundfile as sf
import os

WAVELET = 'db1'  # wavelet type for DWT
DWT_LEVEL = 2  # Decomposition level for DWT
DWT_SCALE = 300  # Scaling factor for coefficient quant

LENGTH_BITS = 32  # no of bits for payload length header
HASH_LEN_BITS = 256  # num of bits for hash

MAX_CAPACITY_BYTES = 10 * 1024  # Maximum 10kb payload

# encryption key - simple number for XOR
ENCRYPTION_KEY = 42


def simple_encrypt(text, key=ENCRYPTION_KEY):
    # simple XOR encryption to make message harder to detect
    encrypted_bytes = []
    for char in text.encode('utf-8'):
        encrypted_bytes.append(char ^ key)  # XOR with key
    # convert back to string that can be stored as bits
    return bytes(encrypted_bytes).decode('latin-1')


def simple_decrypt(encrypted_text, key=ENCRYPTION_KEY):
    # decrypt the XOR encrypted text
    decrypted_bytes = []
    for char in encrypted_text.encode('latin-1'):
        decrypted_bytes.append(char ^ key)  # XOR again to decrypt
    return bytes(decrypted_bytes).decode('utf-8')


def text_to_bits_arr(text):
    byte_data = text.encode('utf-8')
    bit_string = ''.join(f'{byte:08b}' for byte in byte_data)
    return np.array(list(bit_string), dtype=np.int8)


def hexhash_to_bits(hash_hex):
    # hexadecimal hash to array of bits
    int_value = int(hash_hex, 16)
    binary_string = bin(int_value)[2:].zfill(HASH_LEN_BITS)
    return np.array(list(binary_string), dtype=np.int8)


def calculate_capacity(signal_length):
    # maximum capacity based on signal length
    dwt_coeffs = signal_length // (2 ** DWT_LEVEL)
    dwt_capacity_bits = dwt_coeffs - LENGTH_BITS
    dwt_capacity_bytes = dwt_capacity_bits // 8

    capacity_info = {
        'dwt_bytes': dwt_capacity_bytes,
        'max_bytes': min(dwt_capacity_bytes, MAX_CAPACITY_BYTES)
    }
    return capacity_info


def embed_dwt_10kb(signal, payload_bits, hash_bits=None, scale=DWT_SCALE, wavelet=WAVELET, level=DWT_LEVEL):
    #  payload bits in DWT coefficients using leasit sig bit
    print(" Embedding data ")
    payload_length = len(payload_bits)

    #  wavelet decomp
    coeffs = pywt.wavedec(signal.astype(np.float64), wavelet, level=level)
    approximation_coeffs = coeffs[0]

    # use 10kb coz otherwise too much noise
    max_bits = MAX_CAPACITY_BYTES * 8
    if payload_length > max_bits:
        print(f"  10KB limit: {max_bits} bits")
        payload_bits = payload_bits[:max_bits]
        payload_length = max_bits

    # Check for capacity
    if payload_length + LENGTH_BITS > len(approximation_coeffs):
        usable_bits = len(approximation_coeffs) - LENGTH_BITS
        print(f" Signal limited to {usable_bits // 8} bytes")
        payload_bits = payload_bits[:usable_bits]
        payload_length = usable_bits

    # length header plus actal data
    length_binary = f'{payload_length:0{LENGTH_BITS}b}'
    length_bits = np.array(list(length_binary), dtype=np.int8)
    full_payload = np.concatenate((length_bits, payload_bits)).astype(np.int8)

    quantized_coeffs = np.round(approximation_coeffs * scale).astype(np.int64)

    for index, bit_value in enumerate(full_payload):
        quantized_coeffs[index] = (quantized_coeffs[index] & ~1) | int(bit_value)

    # if hash_bits provided, embed them after the main payload as backup
    if hash_bits is not None and len(approximation_coeffs) > len(full_payload) + len(hash_bits):
        hash_start = len(full_payload)
        for i, bit_val in enumerate(hash_bits):
            pos = hash_start + i
            if pos < len(quantized_coeffs):
                quantized_coeffs[pos] = (quantized_coeffs[pos] & ~1) | int(bit_val)
        print(f" Also embedded {len(hash_bits)} hash bits in DWT as backup")

    # reconstructsignal with embedded data
    coeffs[0] = (quantized_coeffs.astype(np.float64) / float(scale))
    stego_signal = pywt.waverec(coeffs, wavelet)

    # check
    if len(stego_signal) > len(signal):
        stego_signal = stego_signal[:len(signal)]
    elif len(stego_signal) < len(signal):
        stego_signal = np.pad(stego_signal, (0, len(signal) - len(stego_signal)))

    stego_signal = np.clip(stego_signal, -1.0, 1.0)
    print(f" Embedded {len(payload_bits)} bits ({len(payload_bits) // 8} bytes)")
    return stego_signal, len(payload_bits)


def embed_fft_magnitude_10kb(signal, hash_bits, gain_factor=3.0):  # much higher gain for reliability
    #  hash bits in FFT magnitude relationships
    print("hash in freq")
    signal_length = len(signal)

    # FFT of the source
    frequency_domain = np.fft.fft(signal)
    half_length = signal_length // 2
    available_pairs = (half_length - 2) // 2

    # check how many bits can be embedded
    bits_to_embed = min(len(hash_bits), available_pairs)

    if bits_to_embed < len(hash_bits):
        hash_bits = hash_bits[:bits_to_embed]

    print(f"Embedding {bits_to_embed} hash bits with gain {gain_factor}")

    # embed each bit  modifying magnitude relationships of frequency pairs
    for bit_index in range(bits_to_embed):
        freq_index1 = 2 + 2 * bit_index
        freq_index2 = freq_index1 + 1

        magnitude1 = np.abs(frequency_domain[freq_index1])
        magnitude2 = np.abs(frequency_domain[freq_index2])
        phase1 = np.angle(frequency_domain[freq_index1])
        phase2 = np.angle(frequency_domain[freq_index2])

        current_bit = int(hash_bits[bit_index])

        # adjust magnitudes to encode the bit - use much larger gain
        if current_bit == 1:
            # make magnitude1 much larger than magnitude2
            target_ratio = gain_factor
            if magnitude1 <= magnitude2 * target_ratio:
                magnitude1 = magnitude2 * target_ratio * 1.5
                magnitude2 = magnitude2 / target_ratio
        else:  # bit 0
            # make magnitude2 much larger than magnitude1
            target_ratio = gain_factor
            if magnitude2 <= magnitude1 * target_ratio:
                magnitude2 = magnitude1 * target_ratio * 1.5
                magnitude1 = magnitude1 / target_ratio

        # chenge frequency components with modified magnitudes
        frequency_domain[freq_index1] = magnitude1 * np.exp(1j * phase1)
        frequency_domain[freq_index2] = magnitude2 * np.exp(1j * phase2)

        # Maintain symmetry
        frequency_domain[-freq_index1] = np.conj(frequency_domain[freq_index1])
        frequency_domain[-freq_index2] = np.conj(frequency_domain[freq_index2])

    #  back to time domain
    stego_signal = np.fft.ifft(frequency_domain).real
    stego_signal = np.clip(stego_signal, -1.0, 1.0)
    print("hash embedding completed")
    return stego_signal


def ensure_cover_exists(file_path, sample_rate=44100, duration=10.0, frequency=440.0):
    # Create a simple sin file
    if os.path.exists(file_path):
        return

    print(f"[IO] Generating original audio: {file_path} ({duration} seconds)")
    time_points = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * time_points)
    sf.write(file_path, sine_wave.astype(np.float32), sample_rate, subtype='FLOAT')


def read_mono_audio(file_path):
    #  convert to mono if necessary
    audio_data, sample_rate = sf.read(file_path, dtype='float64')
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    return audio_data, sample_rate


def write_mono_audio(file_path, audio_data, sample_rate):
    sf.write(file_path, audio_data.astype(np.float32), sample_rate, subtype='FLOAT')


def verify_stego_audio(stego_path, original_text):
    # verify that the stego audio can be decoded correctly
    print("\n verifying stego audio...")

    try:
        # import decoder functions for verification
        from decoder import decode_audio_10kb

        decode_results = decode_audio_10kb(stego_path)

        if decode_results['method'] == 'none' or not decode_results['text']:
            return False, "AUDIO FILE PROVIDED CANNOT BE USED - try another audio file"

        extracted_text = decode_results['text']

        # check if extracted text is gibberish
        printable_count = sum(1 for c in extracted_text if c.isprintable() or c in '\n\r\t')
        printable_ratio = printable_count / len(extracted_text) if extracted_text else 0

        if printable_ratio < 0.7:
            return False, "text is not UTF-8 compatible - try different text or audio"

        # check if extraction matches original
        if extracted_text == original_text:
            return True, "verification passed"

        # check hash verification
        if decode_results['hash_verified']:
            return True, "verification passed"

        return False, "extracted text differs from original...MAKE SURE THE CHARACTES IN THE TEXT CAN BE ENCODED USING UTF-8"

    except Exception as e:
        return False, f"verification error: {str(e)}"


def encode_audio_10kb(cover_path, stego_path, secret_text):
    print("ENCODER")

    # Ensure ORIGINAL audio exists
    ensure_cover_exists(cover_path, duration=10.0)
    cover_audio, sample_rate = read_mono_audio(cover_path)

    print(f" Original file: {cover_path} (length: {len(cover_audio)}, sample rate: {sample_rate})")
    print(f"secret message length: {len(secret_text)} characters")

    # Calculate available capacity
    capacity_info = calculate_capacity(len(cover_audio))
    print(f"\n CAPACITY ANALYSIS:")
    print(f"  - DWT Capacity: {capacity_info['dwt_bytes']} bytes")

    # Check message size and shorten if necessary
    message_size_bytes = len(secret_text.encode('utf-8'))
    if message_size_bytes > MAX_CAPACITY_BYTES:
        print(f"!!!!!!!! message ({message_size_bytes} bytes) exceeds 10KB limit, truncating")
        truncated_text = secret_text[:MAX_CAPACITY_BYTES]
        secret_text = truncated_text
        print(f"Truncated to {len(secret_text)} characters")

    # encrypt the message before embedding
    encrypted_text = simple_encrypt(secret_text)
    print(f"Message encrypted with XOR key {ENCRYPTION_KEY}")

    # Convert encrypted text to binary and compute hash of ORIGINAL text
    message_bits = text_to_bits_arr(encrypted_text)
    message_hash = hashlib.sha256(secret_text.encode('utf-8')).hexdigest()
    hash_bits = hexhash_to_bits(message_hash)
    print(f"Verification hash: {message_hash}")

    print("\n embedding ")

    # Embed message AND hash in DWT domain (hash as backup)
    stego_audio, embedded_bits = embed_dwt_10kb(cover_audio, message_bits, hash_bits)
    # Embed hash in FFT domain with very high gain for reliability
    stego_audio = embed_fft_magnitude_10kb(stego_audio, hash_bits, gain_factor=3.0)

    embedded_bytes = embedded_bits // 8
    print(f"\n Embedded {embedded_bytes} bytes")

    # Save stego audio temporarily for verification - use .wav extension for temp file
    base_name = os.path.splitext(stego_path)[0]
    temp_stego_path = base_name + "_temp.wav"
    write_mono_audio(temp_stego_path, stego_audio, sample_rate)
    print(f" temporary stego audio saved: {temp_stego_path}")

    # Verify extraction works before final save
    verification_ok, verification_msg = verify_stego_audio(temp_stego_path, secret_text)

    if verification_ok:
        # replace temporary file with final file
        if os.path.exists(stego_path):
            os.remove(stego_path)
        os.rename(temp_stego_path, stego_path)
        print(f" verification passed")
        print(f"\n Output: {stego_path}")
        print(" Encoding process completed successfully")
    else:
        # remove temporary file on failure
        if os.path.exists(temp_stego_path):
            os.remove(temp_stego_path)
        print(f" verification failed: {verification_msg}")
        print(" stego audio file was not saved")


if __name__ == '__main__':
    print("ENCODER")

    cover_input = input("enter cover audio path [fall back to: cover.wav]: ").strip()
    if not cover_input:
        cover_input = "cover.wav"

    stego_output = input("Enter output stego audio path [create: stego.wav]: ").strip()
    if not stego_output:
        stego_output = "stego.wav"

    print("\nEnter text:")
    secret_message = input("> ").strip()
    if not secret_message:
        print("Error:nothing typed")
        exit(1)

    try:
        encode_audio_10kb(cover_input, stego_output, secret_message)

        print(f"message length: {len(secret_message)} characters")
        print(f"Message size: {len(secret_message.encode('utf-8'))} bytes")

        print(f"output file: {stego_output}")

    except Exception as error:
        print(f"Error during encoding: {error}")