import os
import whisper
from groq import Groq
import sounddevice as sd
import numpy as np
import tempfile
import wave
from dotenv import load_dotenv
import tkinter as tk
from tkinter import ttk, messagebox
import queue
import time
import pyperclip
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# Load environment variables from .env file
load_dotenv()

# Set up the Whisper and Groq client
whisper_model = whisper.load_model("small.en")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# System prompt (keep the same)
system_prompt = """
You are a helpful assistant designed to convert spoken equations into LaTeX code. 
When a user speaks a mathematical equation, transcribe it into a clean text form, 
then convert the text into a valid LaTeX equation. 
You shall only answer "LaTeX code: whatever code here", don't reply anything else or
any explanations.

For example:
If the user says "A is equal to tan inverse x", your task is to:
1. Transcribe the equation into a clean format: "A = tan^{-1}(x)"
2. Convert the transcribed equation into LaTeX code: "A = \\tan^{-1}(x)"
3. Provide this LaTeX code as output.

Give the correct equations. Don't make mistakes like adding elements after carbon like hydrogen in glucose to subscript for example in chemical equations.

The LaTeX code should be formatted in the standard Latex form.
"""

class AudioRecorder:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.recording = False
        self.audio_data = []
        
    def callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.extend(indata.copy())
            
    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.stream = sd.InputStream(
            callback=self.callback,
            channels=1,
            samplerate=16000,
            dtype='int16'
        )
        self.stream.start()
        
    def stop_recording(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()
        return np.array(self.audio_data)

def text_to_latex(text):
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    
    chat_completion = client.chat.completions.create(
        messages=conversation,
        model="llama3-70b-8192",
    )

    latex_code = chat_completion.choices[0].message.content
    if latex_code.startswith("LaTeX code: "):
        latex_code = latex_code[len("LaTeX code: "):]
    return latex_code

class MainApplication(ttk.Window):
    def __init__(self):
        super().__init__(themename="minty")
        self.title("Equation AI - Speech to LaTeX Converter")
        self.geometry("900x600")
        self.resizable(True, True)
        
        # Initialize variables
        self.recorder = AudioRecorder()
        self.recording = False
        self.start_time = None
        self.transcribed_text = ""
        self.latex_code = ""
        
        # Create UI
        self.create_widgets()
        self.create_bindings()
        
        # Start UI update loop
        self.after(100, self.update_ui)
    
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        # Control Panel
        control_frame = ttk.Labelframe(main_frame, text="Recording Controls", padding=15)
        control_frame.pack(fill=X, pady=5)
        
        self.record_btn = ttk.Button(
            control_frame,
            text="Start Recording",
            command=self.toggle_recording,
            bootstyle=SUCCESS,
            width=15
        )
        self.record_btn.pack(side=LEFT, padx=5)
        
        self.copy_text_btn = ttk.Button(
            control_frame,
            text="Copy Text",
            command=self.copy_text,
            bootstyle=INFO,
            width=10
        )
        self.copy_text_btn.pack(side=RIGHT, padx=5)
        
        self.copy_latex_btn = ttk.Button(
            control_frame,
            text="Copy LaTeX",
            command=self.copy_latex,
            bootstyle=INFO,
            width=10
        )
        self.copy_latex_btn.pack(side=RIGHT, padx=5)
        
        # Status Bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            bootstyle=INFO,
            padding=5,
            anchor=CENTER
        )
        status_bar.pack(fill=X, pady=5)
        
        # Results Panel
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=BOTH, expand=True, pady=10)
        
        # Transcription Display
        trans_label = ttk.Label(results_frame, text="Transcribed Text:")
        trans_label.pack(anchor=NW)
        
        self.trans_text = tk.Text(
            results_frame,
            height=4,
            wrap=WORD,
            font=('Helvetica', 12),
            padx=10,
            pady=10
        )
        self.trans_text.pack(fill=X, pady=5)
        
        # LaTeX Display
        latex_label = ttk.Label(results_frame, text="LaTeX Code:")
        latex_label.pack(anchor=NW)
        
        self.latex_text = tk.Text(
            results_frame,
            height=4,
            wrap=WORD,
            font=('Helvetica', 12),
            padx=10,
            pady=10
        )
        self.latex_text.pack(fill=BOTH, expand=True, pady=5)
    
    def create_bindings(self):
        self.bind("<Control-c>", lambda e: self.copy_text())
        self.bind("<Control-l>", lambda e: self.copy_latex())
        self.bind("<space>", lambda e: self.toggle_recording())
    
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        try:
            self.recorder.start_recording()
            self.recording = True
            self.start_time = time.time()
            self.record_btn.config(text="Stop Recording", bootstyle=DANGER)
            self.status_var.set("Recording...")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {str(e)}")
    
    def stop_recording(self):
        try:
            audio_data = self.recorder.stop_recording()
            self.recording = False
            self.record_btn.config(text="Start Recording", bootstyle=SUCCESS)
            self.status_var.set("Processing audio...")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                with wave.open(tmpfile.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_data.tobytes())
                
                # Transcribe
                result = whisper_model.transcribe(tmpfile.name)
                self.transcribed_text = result['text']
                self.latex_code = text_to_latex(self.transcribed_text)
                
                # Update UI
                self.trans_text.delete(1.0, END)
                self.trans_text.insert(END, self.transcribed_text)
                self.latex_text.delete(1.0, END)
                self.latex_text.insert(END, self.latex_code)
                self.status_var.set("Ready")
            
            os.unlink(tmpfile.name)
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set("Error occurred")
    
    def copy_text(self):
        pyperclip.copy(self.transcribed_text)
        self.status_var.set("Text copied to clipboard!")
    
    def copy_latex(self):
        pyperclip.copy(self.latex_code)
        self.status_var.set("LaTeX copied to clipboard!")
    
    def update_ui(self):
        if self.recording:
            duration = int(time.time() - self.start_time)
            self.status_var.set(f"Recording... {duration} seconds")
        self.after(100, self.update_ui)

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()