from moviepy import VideoFileClip
import os

def convert_video(input_path, output_path):
    try:
        # Check if the output file already exists
        if os.path.exists(output_path):
            overwrite = input(f"File {output_path} already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite != "y":
                print("Conversion canceled.")
                return

        # Load the video file and convert
        with VideoFileClip(input_path) as clip:
            clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            print(f"Conversion successful: {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Input MP4 file path
    input_file = "output_video5_deepseek.mp4"  # Replace with your input file path
    # Output file path (change extension to .mov or .mp4 as needed)
    output_file = "output_video5_deepseek.mp4"  # Replace with your desired output file path

    convert_video(input_file, output_file)