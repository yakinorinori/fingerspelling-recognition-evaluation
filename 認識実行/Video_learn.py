import subprocess
import os

video_directory = 'Video/transformer訓練'  # 動画が保存されているディレクトリパス
video_files = [f for f in os.listdir(video_directory) if f.endswith(('.mp4', '.avi'))]  # 動画ファイルの拡張子を指定

for video_file in video_files:
    video_path = os.path.join(video_directory, video_file)
    # Video_test.py スクリプトを実行
    subprocess.run(['python', 'Video_test.py', video_path], check=True)
