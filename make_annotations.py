import pandas as pd
import os
from tqdm import tqdm

# selected_instruments = ["strings, brass, woodwinds, and mallets"]
TRAIN_DIR = "/media/rpbot/Server Grade Disk 1/ML_datasets/NSYNTH/nsynth-train.jsonwav/nsynth-train/audio"
VAL_DIR = "/media/rpbot/Server Grade Disk 1/ML_datasets/NSYNTH/nsynth-valid.jsonwav/nsynth-valid/audio"
TEST_DIR = "/media/rpbot/Server Grade Disk 1/ML_datasets/NSYNTH/nsynth-test.jsonwav/nsynth-test/audio"


def make_csv_annotations(root_dir, output_file_name):
    allowed_instruments = {"brass", "flute", "mallet", "string"}

    # List to collect file data
    data = []

    # Walk through the directory recursively
    for subdir, dirs, files in os.walk(root_dir):
        for file in tqdm(files, desc="Processing files"):
            # Ignore files starting with "synth_lead"
            if file.lower().startswith("synth_lead"):
                continue

            file_path = os.path.join(subdir, file)
            base_name = os.path.splitext(file)[0]  # Remove file extension

            # Parse the filename assuming the pattern:
            # instrument_source_instrumentName-midiPitch-midiVelocity
            instrument, source_type, rest = base_name.split("_")

            # Process only if the instrument is in the allowed list
            if instrument.lower() in allowed_instruments:
                instrument_name, midi_pitch, midi_velocity = rest.split("-")

                data.append(
                    {
                        "file_path": file_path,
                        "file_name": file,
                        "instrument": instrument,
                        "source_type": source_type,
                        "instrument_name": instrument_name,
                        "midi_pitch": midi_pitch,
                        "midi_velocity": midi_velocity,
                    }
                )

    # Create a pandas DataFrame from the collected data
    df = pd.DataFrame(data)
    print(len(df))
    # Write the DataFrame to a CSV file
    output_csv = f"{output_file_name}.csv"
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    make_csv_annotations(TRAIN_DIR, "train_data_annotations")
    # make_csv_annotations(VAL_DIR, "val_data_annotations")
    # make_csv_annotations(TEST_DIR, "test_data_annotations")
