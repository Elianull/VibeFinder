import whisperx
import numpy as np
import gc
from sklearn.feature_extraction.text import CountVectorizer

# Initialize Count Vectorizer for the transcribed text
vectorizer = CountVectorizer()

def extract_audio_features(audio_file_path):
    """
    Extract features from audio data using WhisperX.

    Parameters:
        audio_file_path (str): The path to the audio file.

    Returns:
        vector (array): The extracted feature vector.
    """
    device = "cuda"  # Assuming you have a compatible GPU
    batch_size = 16  # Reduce if low on GPU memory
    compute_type = "float16"  # Change to "int8" if low on GPU memory (may reduce accuracy)

    # Load the ASR model
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    # Load the audio
    audio = whisperx.load_audio(audio_file_path)

    # Transcribe the audio
    result = model.transcribe(audio, batch_size=batch_size)

    # Align the transcription for word-level timestamps
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # Perform speaker diarization
    diarize_model = whisperx.DiarizationPipeline(device=device)  # Assume HF token is set as an env variable or similar
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Clean up GPU resources if needed
    del model, model_a, diarize_model
    gc.collect()

    # Extract transcriptions, timestamps, and speaker IDs from the result
    transcriptions = [segment['transcription'] for segment in result['segments']]
    timestamps = [segment['start_time'] for segment in result['segments']]
    speaker_ids = [segment['speaker_id'] for segment in result['segments']]

    # Create a "bag-of-words" vector for the transcriptions
    transcription_matrix = vectorizer.fit_transform(transcriptions)
    transcription_vector = transcription_matrix.toarray().flatten()

    # Normalize the timestamps and speaker IDs to fit them into the feature vector
    timestamps = np.array(timestamps) / max(timestamps)
    speaker_ids = np.array(speaker_ids) / max(speaker_ids)

    # Concatenate all these into one feature vector
    feature_vector = np.concatenate([transcription_vector, timestamps, speaker_ids])

    return feature_vector
