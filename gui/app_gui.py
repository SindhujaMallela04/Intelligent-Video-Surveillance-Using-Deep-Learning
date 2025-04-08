# import tkinter as tk
# from tkinter import filedialog, messagebox, scrolledtext
# from tkinter import ttk
# import threading
# import os
# import sys

# # Import the inference function from your model module
# sys.path.append(os.path.abspath("Project"))  # Update this to match your structure
# from core.inference import run_action_recognition


# class SurveillanceApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Intelligent Video Surveillance using Deep Learning")
#         self.root.geometry("700x600")
#         self.root.resizable(False, False)

#         # Video file path
#         self.video_path = tk.StringVar()

#         self.create_widgets()

#     def create_widgets(self):
#         tk.Label(self.root, text="Select a video file for action recognition:", font=("Arial", 12)).pack(pady=10)

#         # Entry to show file path
#         path_entry = tk.Entry(self.root, textvariable=self.video_path, width=60, state="readonly")
#         path_entry.pack(pady=5)

#         # Browse button
#         browse_btn = tk.Button(self.root, text="Browse", command=self.browse_file)
#         browse_btn.pack(pady=5)

#         # Start detection button
#         detect_btn = tk.Button(self.root, text="Start Detection", command=self.start_detection)
#         detect_btn.pack(pady=10)

#         # Video display label
#         self.video_label = ttk.Label(self.root)
#         self.video_label.pack(pady=10)

#         # Log output display
#         tk.Label(self.root, text="Logs:", font=("Arial", 12)).pack(pady=(15, 0))
#         self.log_box = scrolledtext.ScrolledText(self.root, width=80, height=10, state='disabled')
#         self.log_box.pack(pady=5)

#     def browse_file(self):
#         filename = filedialog.askopenfilename(
#             title="Select a Video File",
#             filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
#         )
#         if filename:
#             self.video_path.set(filename)

#     def start_detection(self):
#         if not self.video_path.get():
#             messagebox.showwarning("No file selected", "Please select a video file first.")
#             return

#         self.log("🔍 Running action recognition on selected video...")
#         threading.Thread(target=self.run_detection_thread, daemon=True).start()

#     def run_detection_thread(self):
#         try:
#             run_action_recognition(
#                 video_label=self.video_label,
#                 source=0, #self.video_path.get(),
#                 flip=False,
#                 skip_first_frames=32
#             )
#             self.log("✅ Detection completed.")
#         except Exception as e:
#             self.log(f"❌ Error: {e}")
#             messagebox.showerror("Error", str(e))

#     def log(self, message):
#         self.log_box.configure(state='normal')
#         self.log_box.insert(tk.END, message + "\n")
#         self.log_box.configure(state='disabled')
#         self.log_box.see(tk.END)


# def launch_app():
#     root = tk.Tk()
#     app = SurveillanceApp(root)
#     root.mainloop()


import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import threading
import os
import sys

# Import the inference function from your model module
sys.path.append(os.path.abspath("Project"))  # Update this if needed
from core.inference import run_action_recognition


class SurveillanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Video Surveillance using Deep Learning")
        self.root.geometry("700x600")
        self.root.resizable(False, False)
        # self.running = False

        self.video_path = tk.StringVar()
        self.mode = tk.StringVar(value="Upload Video")  # Default mode

        self.create_widgets()

    def create_widgets(self):
        # Mode Selection
        tk.Label(self.root, text="Select Mode:", font=("Arial", 12)).pack(pady=10)
        mode_dropdown = ttk.Combobox(self.root, textvariable=self.mode, values=["Upload Video", "Live Webcam"], state="readonly", width=20)
        mode_dropdown.pack(pady=5)
        mode_dropdown.bind("<<ComboboxSelected>>", self.on_mode_change)

        # Entry for video path (used in Upload mode)
        self.path_entry = tk.Entry(self.root, textvariable=self.video_path, width=60, state="readonly")
        self.path_entry.pack(pady=5)

        # Browse button
        self.browse_btn = tk.Button(self.root, text="Browse", command=self.browse_file)
        self.browse_btn.pack(pady=5)

        # Start detection button
        detect_btn = tk.Button(self.root, text="Start Detection", command=self.start_detection)
        detect_btn.pack(pady=10)

        # Stop detection
        # self.stop_btn = tk.Button(self.root, text="Stop Detection", command=self.stop_detection, state='disabled')
        # self.stop_btn.pack(pady=5)

        # Video display label
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(pady=10)

        # Logs
        tk.Label(self.root, text="Logs:", font=("Arial", 12)).pack(pady=(15, 0))
        self.log_box = scrolledtext.ScrolledText(self.root, width=80, height=10, state='disabled')
        self.log_box.pack(pady=5)

    def on_mode_change(self, event=None):
        mode = self.mode.get()
        if mode == "Live Webcam":
            self.video_path.set("")
            self.path_entry.configure(state="disabled")
            self.browse_btn.configure(state="disabled")
        else:
            self.path_entry.configure(state="normal")
            self.browse_btn.configure(state="normal")

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if filename:
            self.video_path.set(filename)

    def start_detection(self):
        if self.mode.get() == "Upload Video" and not self.video_path.get():
            messagebox.showwarning("No file selected", "Please select a video file first.")
            return

        self.log("🔍 Starting action recognition...")
        # self.running = True
        # self.stop_btn.config(state='normal')
        threading.Thread(target=self.run_detection_thread, daemon=True).start()

    # def stop_detection(self):
    #     self.log("🛑 Stopping detection...")
    #     self.running = False
    #     self.stop_btn.config(state='disabled')

    def run_detection_thread(self):
        try:
            source = 0 if self.mode.get() == "Live Webcam" else self.video_path.get()
            run_action_recognition(
                video_label=self.video_label,
                source=0,
                flip=(self.mode.get() == "Live Webcam"),  # Flip only for webcam
                skip_first_frames=32 if self.mode.get() == "Upload Video" else 0,
                # should_stop=lambda: not self.running
            )
            self.log("✅ Detection completed.")
            # self.running = False
            # self.stop_btn.config(state='disabled')
        except Exception as e:
            self.log(f"❌ Error: {e}")
            messagebox.showerror("Error", str(e))
            # self.running = False
            # self.stop_btn.config(state='disabled')

    def log(self, message):
        self.log_box.configure(state='normal')
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.configure(state='disabled')
        self.log_box.see(tk.END)


def launch_app():
    root = tk.Tk()
    app = SurveillanceApp(root)
    root.mainloop()






