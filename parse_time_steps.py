from pydub import AudioSegment
import os
import pandas as pd
import numpy as np

def parse_timestamp(timestamp):
    """Convert a timestamp string (HH:MM:SS or MM:SS) to milliseconds."""
    if pd.isna(timestamp):
        return 0  # Default to 0 if the timestamp is NaN
    if isinstance(timestamp, (float, int)):
        timestamp = str(timestamp)  # Convert numbers to strings
    parts = list(map(int, timestamp.split(":")))
    if len(parts) == 2:  # MM:SS
        minutes, seconds = parts
        return (minutes * 60 + seconds) * 1000
    elif len(parts) == 3:  # HH:MM:SS
        hours, minutes, seconds = parts
        return (hours * 3600 + minutes * 60 + seconds) * 1000
    return 0

def split_audio_by_timestamp_incremental(df, audio_folder="audio", output_csv="output_segments.csv", output_folder="output"):
    """Split audio files by timestamp and save the segments, updating the CSV incrementally."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_csv):
        pd.DataFrame(columns=df.columns.tolist() + ['audio_segment_path']).to_csv(output_csv, index=False)

    for idx, row in df.iterrows():
        wav_filename = row['wav_files']
        if pd.isna(wav_filename) or not isinstance(wav_filename, str) or not wav_filename.strip():
            row['audio_segment_path'] = ""
            pd.DataFrame([row]).to_csv(output_csv, mode='a', header=False, index=False)
            print(f"Skipping row {idx + 1} due to missing or invalid 'wav_filename', added with empty 'audio_segment_path'")
            continue
        audio_file = f"{audio_folder}/{wav_filename}"
        start_time = parse_timestamp(row['start'])
        end_time = (
            parse_timestamp(df.iloc[idx + 1]['start'])
            if idx + 1 < len(df) and df.iloc[idx + 1]['wav_files'] == wav_filename
            else None
        )

        try:
            audio = AudioSegment.from_wav(audio_file)
            if end_time is None:
                end_time = len(audio)
            segment = audio[start_time:end_time]
            segment_filename = f"{output_folder}/{wav_filename}_segment_{idx + 1}.wav"
            segment.export(segment_filename, format="wav")
            row['audio_segment_path'] = segment_filename
            pd.DataFrame([row]).to_csv(output_csv, mode='a', header=False, index=False)
            print(f"Processed audio segment {idx + 1} and saved to {segment_filename}")
        except Exception as e:
            print(f"Error processing file {audio_file} at segment {idx + 1}: {e}")
            row['audio_segment_path'] = ""
            pd.DataFrame([row]).to_csv(output_csv, mode='a', header=False, index=False)

df = pd.read_csv('jakarta_field_station_with_wav.csv')
output_csv = "jakarta_field_station_timestamp_with_audio.csv"
output_folder = "audio_cropped"
split_audio_by_timestamp_incremental(df, audio_folder=".", output_csv=output_csv, output_folder=output_folder)

print(f"Processed segments are saved in {output_csv}")