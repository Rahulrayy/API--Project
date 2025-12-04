import numpy as np
import pywt
import hashlib
import os
import soundfile as sf

WAVELET = 'db1'
DWT_LEVEL = 2
DWT_SCALE = 300

LENGTH_BITS = 32
HASH_LEN_BITS = 256

ENCRYPTION_KEY = 42


def simple_decrypt(encrypted_text, key=ENCRYPTION_KEY):
    decrypted_bytes = []
    for char in encrypted_text.encode('latin-1'):
        decrypted_bytes.append(char ^ key)
    return bytes(decrypted_bytes).decode('utf-8')


def bits_array_to_text(bits_array):
    if len(bits_array) == 0:
        return ''
    binary_string = ''.join(str(int(bit)) for bit in bits_array)
    complete_bytes = (len(binary_string) // 8) * 8
    binary_string = binary_string[:complete_bytes]
    if len(binary_string) == 0:
        return ''
    try:
        byte_data = bytearray(int(binary_string[i:i + 8], 2)
                              for i in range(0, len(binary_string), 8))
        return bytes(byte_data)
    except Exception:
        return ''


def bits_to_hexadecimal(bits_array):
    binary_string = ''.join(str(int(bit)) for bit in bits_array)
    if len(binary_string) < HASH_LEN_BITS:
        binary_string = binary_string.zfill(HASH_LEN_BITS)
    elif len(binary_string) > HASH_LEN_BITS:
        binary_string = binary_string[:HASH_LEN_BITS]
    integer_value = int(binary_string, 2)
    return f'{integer_value:064x}'


def extract_dwt_10kb(signal, scale=DWT_SCALE, wavelet=WAVELET, level=DWT_LEVEL, extract_hash=False):
    print(" extracting data from wavelet coefficients")
    coeffs = pywt.wavedec(signal.astype(np.float64), wavelet, level=level)
    approximation_coeffs = coeffs[0]
    quantized_coeffs = np.round(approximation_coeffs * scale).astype(np.int64)
    if len(quantized_coeffs) < LENGTH_BITS:
        return None, 0, None
    length_binary = ''.join(str(int((quantized_coeffs[i] & 1)))
                            for i in range(LENGTH_BITS))
    try:
        payload_length = int(length_binary, 2)
    except:
        return None, 0, None
    if payload_length <= 0 or payload_length > len(quantized_coeffs) - LENGTH_BITS:
        return None, 0, None
    extracted_bits = np.array([int((quantized_coeffs[i] & 1))
                               for i in range(LENGTH_BITS, LENGTH_BITS + payload_length)],
                              dtype=np.int8)
    hash_bits = None
    if extract_hash:
        hash_start = LENGTH_BITS + payload_length
        if hash_start + HASH_LEN_BITS <= len(quantized_coeffs):
            hash_bits = np.array([int((quantized_coeffs[i] & 1))
                                  for i in range(hash_start, hash_start + HASH_LEN_BITS)],
                                 dtype=np.int8)
            print(f" Also extracted {len(hash_bits)} hash bits from DWT backup")
    print(f" extracted {len(extracted_bits)} bits ({len(extracted_bits) // 8} bytes)")
    return extracted_bits, payload_length, hash_bits


def extract_fft_magnitude_10kb(signal, expected_bits):
    print(" extracting verification hash from frequency domain")
    signal_length = len(signal)
    frequency_domain = np.fft.fft(signal)
    half_length = signal_length // 2
    available_pairs = (half_length - 2) // 2
    bits_to_extract = min(expected_bits, available_pairs)
    if bits_to_extract <= 0:
        return np.array([], dtype=np.int8)
    extracted_bits = []
    for bit_index in range(bits_to_extract):
        freq_index1 = 2 + 2 * bit_index
        freq_index2 = freq_index1 + 1
        magnitude1 = np.abs(frequency_domain[freq_index1])
        magnitude2 = np.abs(frequency_domain[freq_index2])
        ratio = magnitude1 / (magnitude2 + 1e-12)
        if ratio > 2.0:
            bit_value = 1
        elif ratio < 0.5:
            bit_value = 0
        else:
            bit_value = 1 if magnitude1 > magnitude2 else 0
        extracted_bits.append(bit_value)
    hash_bits = np.array(extracted_bits, dtype=np.int8)
    print(f" extracted {len(hash_bits)} hash bits from FFT")
    return hash_bits


def read_mono_audio(file_path):
    try:
        audio_data, sample_rate = sf.read(file_path, dtype='float64')
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        return audio_data, sample_rate
    except Exception as error:
        print(f"cannot reading audio file: {error}")
        raise


def decode_audio_10kb(stego_path):
    print("DECODER")
    try:
        stego_audio, sample_rate = read_mono_audio(stego_path)
        print(f" File {stego_path} (length: {len(stego_audio)}, sample rate: {sample_rate})")
    except Exception as error:
        return {
            'method': 'none',
            'text': '',
            'hash_verified': False,
            'extracted_hash': '',
            'error': str(error)
        }
    extracted_hash_bits_fft = extract_fft_magnitude_10kb(stego_audio, HASH_LEN_BITS)
    extracted_hash_hex_fft = bits_to_hexadecimal(extracted_hash_bits_fft)
    print(f" FFT verification hash: {extracted_hash_hex_fft}")
    print("\n  DWT extraction")
    message_bits, message_length, extracted_hash_bits_dwt = extract_dwt_10kb(stego_audio, extract_hash=True)
    if extracted_hash_bits_dwt is not None:
        extracted_hash_hex = bits_to_hexadecimal(extracted_hash_bits_dwt)
        hash_source = "DWT backup"
    else:
        extracted_hash_hex = extracted_hash_hex_fft
        hash_source = "FFT"
    print(f" Using {hash_source} verification hash: {extracted_hash_hex}")
    if message_bits is not None and message_length > 0:
        encrypted_bytes = bits_array_to_text(message_bits)
        if encrypted_bytes:
            try:
                encrypted_text = encrypted_bytes.decode('latin-1')
            except:
                encrypted_text = encrypted_bytes.decode('latin-1', errors='ignore')
            try:
                decrypted_text = simple_decrypt(encrypted_text)
                print(f" Decrypted message with XOR key {ENCRYPTION_KEY}")
            except Exception as e:
                print(f" Decryption failed: {e}")
                try:
                    decrypted_text = encrypted_bytes.decode('utf-8', errors='ignore')
                    print(" Used raw bytes (no decryption)")
                except:
                    decrypted_text = ""
            if decrypted_text:
                computed_hash = hashlib.sha256(decrypted_text.encode('utf-8')).hexdigest()
                print(f" Computed hash from decrypted text: {computed_hash}")
                if extracted_hash_hex == computed_hash:
                    print(" Extraction successful and hash verified")
                    return {
                        'method': 'DWT',
                        'text': decrypted_text,
                        'hash_verified': True,
                        'extracted_hash': extracted_hash_hex,
                        'error': None
                    }
                else:
                    print("Hash verification failed but message extracted")
                    print(f" Expected: {extracted_hash_hex}")
                    print(f" Got: {computed_hash}")
                    return {
                        'method': 'DWT',
                        'text': decrypted_text,
                        'hash_verified': False,
                        'extracted_hash': extracted_hash_hex,
                        'error': 'Hash verification failed but message extracted'
                    }
    print("\nMessage extraction failed")
    return {
        'method': 'none',
        'text': '',
        'hash_verified': False,
        'extracted_hash': extracted_hash_hex,
        'error': 'No valid message found in audio'
    }


def save_text_to_file(text, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f" Text saved to: {filename}")
        print(f" Saved {len(text)} characters with all line breaks preserved")
        return True
    except Exception as error:
        print(f" Error saving text file: {error}")
        return False


if __name__ == '__main__':
    print("DECODER")
    stego_input = input("Enter stego audio path [fall back to: stego.wav]: ").strip()
    if not stego_input:
        stego_input = "stego.wav"
    print("\n")
    try:
        decode_results = decode_audio_10kb(stego_input)
        print(f"Extraction Method: {decode_results['method']}")
        print(f"Hash Verified: {decode_results['hash_verified']}")
        if decode_results['text']:
            line_count = decode_results['text'].count('\n') + 1
            char_count = len(decode_results['text'])
            paragraph_count = decode_results['text'].count('\n\n') + 1
            print(f"extracted text length: {char_count} characters")
            print(f"number of lines: {line_count}")
            print(f"number of paragraphs: {paragraph_count}")
            text_preview = decode_results['text']
            preview_length = min(10000, len(text_preview))
            print(f"extracted text preview (first {preview_length} characters):")
            print(text_preview[:preview_length])
            if len(text_preview) > preview_length:
                print(f"... (and {len(text_preview) - preview_length} more characters)")
            print(f"extracted data size: {len(decode_results['text'].encode('utf-8'))} bytes")
            save_choice = input("\nsave extracted text to file? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                base_name = os.path.splitext(stego_input)[0]
                txt_filename = f"{base_name}_extracted.txt"
                custom_name = input(f"enter filename [default: {txt_filename}]: ").strip()
                if custom_name:
                    if not custom_name.endswith('.txt'):
                        custom_name += '.txt'
                    txt_filename = custom_name
                if save_text_to_file(decode_results['text'], txt_filename):
                    try:
                        with open(txt_filename, 'r', encoding='utf-8') as f:
                            saved_content = f.read()
                        if saved_content == decode_results['text']:
                            print(" File verification: all content saved correctly")
                            print(f" Verified: {len(saved_content)} characters saved")
                        else:
                            print(f" Warning: saved file content differs from extracted text")
                            print(f" Extracted: {len(decode_results['text'])} characters")
                            print(f" Saved: {len(saved_content)} characters")
                    except Exception as e:
                        print(f" Could not verify saved file: {e}")
        else:
            print("no message found")
        print(f"Extracted Hash: {decode_results['extracted_hash']}")
        if decode_results.get('error'):
            print(f"error: {decode_results['error']}")
    except Exception as fatal_error:
        print(f" error during decoding: {fatal_error}")
