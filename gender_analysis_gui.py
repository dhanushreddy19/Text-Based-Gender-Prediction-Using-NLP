import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from textblob import TextBlob
import joblib
from datetime import datetime
import os
import re
from tkinter import simpledialog

class GenderAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Analysis Tool")
        self.root.geometry("900x1000")
        self.root.configure(bg='#f0f0f0')
        
        # Configure styles
        self.configure_styles()
        
        # Load models
        self.load_models()
        self.create_gui()
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-a>', lambda e: self.analyze_text())
        self.root.bind('<Control-c>', lambda e: self.clear_all())
        self.root.bind('<Control-s>', lambda e: self.save_history())

    def configure_styles(self):
        style = ttk.Style()
        style.configure('TLabel', font=('Arial', 11))
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Accent.TButton', font=('Arial', 11, 'bold'))
        style.configure('Secondary.TButton', font=('Arial', 10))
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabelframe', background='#f0f0f0')
        style.configure('TLabelframe.Label', font=('Arial', 11, 'bold'))

    def load_models(self):
        try:
            model_path = 'gender_model.joblib'
            vectorizer_path = 'vectorizer.joblib'
            
            if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                raise FileNotFoundError("Model files not found")
                
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print("Models loaded successfully!")
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load models: {str(e)}")
            self.root.destroy()

    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create tooltip manager
        self.tooltips = {}

        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Text Input", padding="10")
        input_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        
        ttk.Label(input_frame, text="Enter Text for Analysis:", style='Header.TLabel').grid(row=0, column=0, pady=(0, 5), sticky='w')
        self.text_input = scrolledtext.ScrolledText(input_frame, width=80, height=5, font=('Arial', 10))
        self.text_input.grid(row=1, column=0, pady=(0, 10))
        
        # Button frame
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=2, column=0, sticky='ew', pady=(0, 5))
        
        analyze_btn = ttk.Button(button_frame, text="Analyze Text (Ctrl+A)", command=self.analyze_text, style='Accent.TButton')
        analyze_btn.grid(row=0, column=0, padx=5)
        self.add_tooltip(analyze_btn, "Analyze the text for gender, sentiment, and mood")
        
        clear_btn = ttk.Button(button_frame, text="Clear (Ctrl+C)", command=self.clear_all, style='Secondary.TButton')
        clear_btn.grid(row=0, column=1, padx=5)
        self.add_tooltip(clear_btn, "Clear input and results")
        
        save_btn = ttk.Button(button_frame, text="Save History (Ctrl+S)", command=self.save_history, style='Secondary.TButton')
        save_btn.grid(row=0, column=2, padx=5)
        self.add_tooltip(save_btn, "Save analysis history to file")

        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=1, column=0, sticky='ew', pady=(0, 20))

        # Create results grid
        results = [
            ("Gender Prediction:", "gender_var"),
            ("Confidence:", "confidence_var"),
            ("Sentiment:", "sentiment_var"),
            ("Polarity:", "polarity_var"),
            ("Mood:", "mood_var")
        ]
        
        for i, (label, var_name) in enumerate(results):
            ttk.Label(results_frame, text=label, style='TLabel').grid(row=i, column=0, pady=5, sticky='w')
            setattr(self, var_name, tk.StringVar())
            ttk.Label(results_frame, textvariable=getattr(self, var_name), style='TLabel').grid(row=i, column=1, pady=5, sticky='w')

        # History section
        history_frame = ttk.LabelFrame(main_frame, text="Analysis History", padding="10")
        history_frame.grid(row=2, column=0, sticky='ew', pady=(0, 10))

        self.history_text = scrolledtext.ScrolledText(history_frame, width=80, height=15, font=('Arial', 10))
        self.history_text.grid(row=0, column=0, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(history_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=1, column=0, sticky='ew', pady=(5, 0))

    def add_tooltip(self, widget, text):
        def show_tooltip(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            label = ttk.Label(self.tooltip, text=text, justify='left',
                            background="#ffffe0", relief='solid', borderwidth=1)
            label.pack()
        
        def hide_tooltip(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
        
        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)

    def preprocess_text(self, text):
        """Preprocess the input text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def analyze_mood(self, text, polarity, subjectivity):
        """Analyze the mood based on text content and sentiment metrics."""
        # Count various indicators
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        word_count = len(text.split())
        
        # Enhanced mood analysis
        if polarity > 0.7 and exclamation_count > 1:
            return "Very Excited/Enthusiastic"
        elif polarity > 0.5 and exclamation_count > 0:
            return "Excited/Enthusiastic"
        elif polarity > 0.3:
            return "Happy/Positive"
        elif polarity < -0.7:
            return "Very Angry/Frustrated"
        elif polarity < -0.5:
            return "Angry/Frustrated"
        elif polarity < -0.3:
            return "Sad/Negative"
        elif question_count > 2:
            return "Very Curious/Questioning"
        elif question_count > 0:
            return "Curious/Questioning"
        elif caps_ratio > 0.5:
            return "Very Emphatic/Intense"
        elif caps_ratio > 0.3:
            return "Emphatic/Intense"
        elif subjectivity > 0.9:
            return "Very Emotional"
        elif subjectivity > 0.8:
            return "Emotional"
        elif subjectivity < 0.1:
            return "Very Objective/Factual"
        elif subjectivity < 0.2:
            return "Objective/Factual"
        else:
            return "Neutral"

    def analyze_text(self):
        """Analyze the input text for gender, sentiment, and mood."""
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to analyze.")
            return

        try:
            # Show progress
            self.progress_var.set(20)
            self.root.update_idletasks()
            
            # Preprocess text
            text = self.preprocess_text(text)
            
            # Gender prediction
            self.progress_var.set(40)
            text_vectorized = self.vectorizer.transform([text])
            gender = self.model.predict(text_vectorized)[0]
            probabilities = self.model.predict_proba(text_vectorized)[0]
            confidence = max(probabilities) * 100

            # Sentiment analysis
            self.progress_var.set(60)
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Determine sentiment label
            if polarity > 0.3:
                sentiment = "Very Positive"
            elif polarity > 0:
                sentiment = "Positive"
            elif polarity < -0.3:
                sentiment = "Very Negative"
            elif polarity < 0:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            # Determine mood
            self.progress_var.set(80)
            mood = self.analyze_mood(text, polarity, subjectivity)

            # Update GUI
            self.gender_var.set(f"{gender.title()}")
            self.confidence_var.set(f"{confidence:.2f}%")
            self.sentiment_var.set(sentiment)
            self.polarity_var.set(f"{polarity:.2f} (Subjectivity: {subjectivity:.2f})")
            self.mood_var.set(mood)

            # Add to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_entry = f"[{timestamp}]\n"
            history_entry += f"Text: {text}\n"
            history_entry += f"Gender: {gender.title()} ({confidence:.2f}%)\n"
            history_entry += f"Sentiment: {sentiment} (Polarity: {polarity:.2f})\n"
            history_entry += f"Mood: {mood}\n"
            history_entry += "-" * 50 + "\n"
            
            self.history_text.insert(tk.END, history_entry)
            self.history_text.see(tk.END)
            
            # Complete progress
            self.progress_var.set(100)
            self.root.after(1000, lambda: self.progress_var.set(0))

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.progress_var.set(0)

    def clear_all(self):
        """Clear all input and results."""
        self.text_input.delete("1.0", tk.END)
        self.gender_var.set("")
        self.confidence_var.set("")
        self.sentiment_var.set("")
        self.polarity_var.set("")
        self.mood_var.set("")
        self.progress_var.set(0)

    def save_history(self):
        """Save the analysis history to a file."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Analysis History"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.history_text.get("1.0", tk.END))
                messagebox.showinfo("Success", "History saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save history: {str(e)}")

def main():
    root = tk.Tk()
    app = GenderAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()