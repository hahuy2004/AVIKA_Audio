import os
import sys
import subprocess

def convert_video(input_path, output_path):
    command = [
        "ffmpeg",
        "-y",
        "-err_detect", "ignore_err",
        "-i", input_path,
        "-vf", "yadif,scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ac", "2",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Convert thành công: {output_path}")
    except subprocess.CalledProcessError:
        print(f"Bỏ qua file lỗi: {input_path}")


def convert_folder(input_folder, output_folder):
    if not os.path.exists(input_folder):
        print(f"Folder input không tồn tại: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):

        input_path = os.path.join(input_folder, filename)

        # bỏ qua folder
        if not os.path.isfile(input_path):
            continue

        name_without_ext = os.path.splitext(filename)[0]
        output_filename = name_without_ext + ".mp4"
        output_path = os.path.join(output_folder, output_filename)

        print(f"Đang convert: {filename}")
        convert_video(input_path, output_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_video.py input_folder output_folder")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    convert_folder(input_folder, output_folder)

# import os
# import sys
# import subprocess

# def convert_video(input_path, output_path):
#     if not os.path.exists(input_path):
#         print(f"File không tồn tại: {input_path}")
#         return

#     command = [
#         "ffmpeg",
#         "-y",
#         "-err_detect", "ignore_err",
#         "-i", input_path,
#         "-vf", "yadif,scale=trunc(iw/2)*2:trunc(ih/2)*2",
#         "-c:v", "libx264",
#         "-preset", "fast",
#         "-pix_fmt", "yuv420p",
#         "-c:a", "aac",          # encode lại audio
#         "-b:a", "192k",         # bitrate audio
#         "-movflags", "+faststart",
#         output_path
#     ]

#     try:
#         subprocess.run(command, check=True)
#         print(f"Convert thành công: {output_path}")
#     except subprocess.CalledProcessError:
#         print("Lỗi khi convert video")

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python convert_video.py input_video output_video")
#         sys.exit(1)

#     convert_video(sys.argv[1], sys.argv[2])