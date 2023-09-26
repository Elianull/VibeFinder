from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os

# Custom modules for feature extraction and utilities
from feature_extraction import (
    extract_text_features, extract_image_features, extract_audio_features,
    extract_excel_features, extract_video_features, extract_source_code_features,
    extract_exe_features
)
from utils.file_utils import (
    read_text_file, read_image_file, read_audio_file, read_excel_file,
    read_video_file, read_source_code_file, read_exe_file
)
from database.vector_db import update_vector, delete_vector
from config.settings import WATCH_DIRECTORY

class Handler(FileSystemEventHandler):
    def process(self, event):
        file_path = event.src_path
        file_extension = os.path.splitext(file_path)[1].lower()

        # Handle file create or modify events
        if event.event_type in ['created', 'modified']:
            vector = None

            # Text files
            if file_extension in ['.pdf', '.docx', '.txt']:
                text = read_text_file(file_path, file_extension)
                vector = extract_text_features(text)
                
            # Image files
            elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                image_data = read_image_file(file_path, file_extension)
                vector = extract_image_features(image_data)

            # Audio files
            elif file_extension in ['.mp3', '.wav']:
                audio = read_audio_file(file_path, file_extension)
                vector = extract_audio_features(audio)
            
            # Excel files
            elif file_extension in ['.xlsx', '.xls']:
                data = read_excel_file(file_path, file_extension)
                vector = extract_excel_features(data)
            
            # Video files
            elif file_extension in ['.mp4', '.avi']:
                video = read_video_file(file_path, file_extension)
                vector = extract_video_features(video)
            
            # Source code files
            elif file_extension in ['.py', '.java', '.cpp', '.h']:
                code = read_source_code_file(file_path, file_extension)
                vector = extract_source_code_features(code)
            
            # Executables
            elif file_extension in ['.exe', '.dll']:
                binary_data = read_exe_file(file_path, file_extension)
                vector = extract_exe_features(binary_data)

            # Update vector database if vector is extracted
            if vector is not None:
                update_vector(file_path, vector)

        # Handle file delete events
        elif event.event_type == 'deleted':
            delete_vector(file_path)

    # Watchdog event hooks
    def on_modified(self, event):
        self.process(event)

    def on_created(self, event):
        self.process(event)

    def on_deleted(self, event):
        self.process(event)

if __name__ == "__main__":
    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIRECTORY, recursive=True)
    observer.start()

    print(f"VibeFinder is now watching {WATCH_DIRECTORY}...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
