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
        
        # Define modern color scheme
        self.colors = {
            'primary': '#2196F3',      # Blue
            'secondary': '#4CAF50',    # Green
            'accent': '#FF9800',       # Orange
            'background': '#F5F5F5',   # Light gray
            'surface': '#FFFFFF',      # White
            'text': '#212121',         # Dark gray
            'text_secondary': '#757575', # Medium gray
            'border': '#E0E0E0',       # Light border
            'success': '#4CAF50',      # Green
            'warning': '#FFC107',      # Yellow
            'error': '#F44336',        # Red
            'info': '#2196F3'          # Blue
        }
        
        # Configure root window
        self.root.configure(bg=self.colors['background'])
        
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
        
        # Configure base styles
        style.configure('TFrame', background=self.colors['background'])
        style.configure('TLabelframe', 
                       background=self.colors['surface'],
                       bordercolor=self.colors['border'],
                       relief='solid',
                       borderwidth=1)
        style.configure('TLabelframe.Label', 
                       font=('Segoe UI', 11, 'bold'),
                       foreground=self.colors['primary'],
                       background=self.colors['surface'])
        
        # Configure labels
        style.configure('TLabel', 
                       font=('Segoe UI', 10),
                       background=self.colors['surface'],
                       foreground=self.colors['text'])
        style.configure('Header.TLabel', 
                       font=('Segoe UI', 12, 'bold'),
                       foreground=self.colors['primary'])
        style.configure('Result.TLabel', 
                       font=('Segoe UI', 10, 'bold'),
                       foreground=self.colors['primary'])
        
        # Configure buttons
        style.configure('Accent.TButton', 
                       font=('Segoe UI', 10, 'bold'),
                       background=self.colors['primary'],
                       foreground='white',
                       padding=5)
        style.configure('Secondary.TButton', 
                       font=('Segoe UI', 10),
                       background=self.colors['secondary'],
                       foreground='white',
                       padding=5)
        
        # Configure progress bar
        style.configure('TProgressbar', 
                       background=self.colors['primary'],
                       troughcolor=self.colors['border'],
                       thickness=10)

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
        
        # Welcome message
        welcome_label = ttk.Label(input_frame, 
                                text="Welcome to Text Analysis Tool!\nEnter your text below to analyze gender, sentiment, and mood.",
                                style='Header.TLabel',
                                justify='center')
        welcome_label.grid(row=0, column=0, pady=(0, 10))
        
        ttk.Label(input_frame, text="Enter Text for Analysis:", style='Header.TLabel').grid(row=1, column=0, pady=(0, 5), sticky='w')
        
        # Text input with placeholder
        self.text_input = scrolledtext.ScrolledText(input_frame, 
                                                  width=80, 
                                                  height=5, 
                                                  font=('Segoe UI', 10),
                                                  bg=self.colors['surface'],
                                                  fg=self.colors['text'],
                                                  insertbackground=self.colors['primary'],
                                                  relief='solid',
                                                  borderwidth=1)
        self.text_input.grid(row=2, column=0, pady=(0, 10))
        self.text_input.insert('1.0', "Type or paste your text here...")
        self.text_input.bind('<FocusIn>', self.clear_placeholder)
        self.text_input.bind('<FocusOut>', self.restore_placeholder)
        
        # Button frame
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=3, column=0, sticky='ew', pady=(0, 5))
        
        analyze_btn = ttk.Button(button_frame, 
                               text="Analyze Text (Ctrl+A)", 
                               command=self.analyze_text, 
                               style='Accent.TButton')
        analyze_btn.grid(row=0, column=0, padx=5)
        self.add_tooltip(analyze_btn, "Analyze the text for gender, sentiment, and mood")
        
        clear_btn = ttk.Button(button_frame, 
                             text="Clear (Ctrl+C)", 
                             command=self.clear_all, 
                             style='Secondary.TButton')
        clear_btn.grid(row=0, column=1, padx=5)
        self.add_tooltip(clear_btn, "Clear input and results")
        
        save_btn = ttk.Button(button_frame, 
                            text="Save History (Ctrl+S)", 
                            command=self.save_history, 
                            style='Secondary.TButton')
        save_btn.grid(row=0, column=2, padx=5)
        self.add_tooltip(save_btn, "Save analysis history to file")

        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=1, column=0, sticky='ew', pady=(0, 20))

        # Create results grid with enhanced styling
        results = [
            ("Gender Prediction:", "gender_var", "👤"),
            ("Confidence:", "confidence_var", "📊"),
            ("Sentiment:", "sentiment_var", "😊"),
            ("Polarity:", "polarity_var", "⚖️"),
            ("Mood:", "mood_var", "🎭")
        ]
        
        for i, (label, var_name, emoji) in enumerate(results):
            ttk.Label(results_frame, 
                     text=f"{emoji} {label}", 
                     style='TLabel').grid(row=i, column=0, pady=5, sticky='w')
            setattr(self, var_name, tk.StringVar())
            result_label = ttk.Label(results_frame, 
                                   textvariable=getattr(self, var_name),
                                   style='Result.TLabel')
            result_label.grid(row=i, column=1, pady=5, sticky='w')

        # History section
        history_frame = ttk.LabelFrame(main_frame, text="Analysis History", padding="10")
        history_frame.grid(row=2, column=0, sticky='ew', pady=(0, 10))

        self.history_text = scrolledtext.ScrolledText(history_frame, 
                                                    width=80, 
                                                    height=15, 
                                                    font=('Segoe UI', 10),
                                                    bg=self.colors['surface'],
                                                    fg=self.colors['text'],
                                                    relief='solid',
                                                    borderwidth=1)
        self.history_text.grid(row=0, column=0, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(history_frame, 
                                          variable=self.progress_var, 
                                          maximum=100,
                                          style='TProgressbar')
        self.progress_bar.grid(row=1, column=0, sticky='ew', pady=(5, 0))
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(history_frame, 
                                    textvariable=self.status_var,
                                    style='TLabel')
        self.status_label.grid(row=2, column=0, sticky='ew', pady=(5, 0))

    def clear_placeholder(self, event):
        if self.text_input.get("1.0", "end-1c") == "Type or paste your text here...":
            self.text_input.delete("1.0", "end")
            self.text_input.configure(fg=self.colors['text'])

    def restore_placeholder(self, event):
        if not self.text_input.get("1.0", "end-1c").strip():
            self.text_input.insert("1.0", "Type or paste your text here...")
            self.text_input.configure(fg=self.colors['text_secondary'])

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
        
        # Enhanced mood analysis with emojis
        if polarity > 0.7 and exclamation_count > 1:
            return "Very Excited/Enthusiastic 🎉"
        elif polarity > 0.5 and exclamation_count > 0:
            return "Excited/Enthusiastic 😃"
        elif polarity > 0.3:
            return "Happy/Positive 😊"
        elif polarity < -0.7:
            return "Very Angry/Frustrated 😡"
        elif polarity < -0.5:
            return "Angry/Frustrated 😠"
        elif polarity < -0.3:
            return "Sad/Negative 😔"
        elif question_count > 2:
            return "Very Curious/Questioning 🤔"
        elif question_count > 0:
            return "Curious/Questioning ❓"
        elif caps_ratio > 0.5:
            return "Very Emphatic/Intense 💪"
        elif caps_ratio > 0.3:
            return "Emphatic/Intense 💪"
        elif subjectivity > 0.9:
            return "Very Emotional 😭"
        elif subjectivity > 0.8:
            return "Emotional 😢"
        elif subjectivity < 0.1:
            return "Very Objective/Factual 📊"
        elif subjectivity < 0.2:
            return "Objective/Factual 📈"
        else:
            return "Neutral 😐"

    def analyze_text(self):
        """Analyze the input text for gender, sentiment, and mood."""
        text = self.text_input.get("1.0", tk.END).strip()
        if not text or text == "Type or paste your text here...":
            messagebox.showwarning("Warning", "Please enter some text to analyze.")
            return

        try:
            # Update status
            self.status_var.set("Analyzing text...")
            self.progress_var.set(20)
            self.root.update_idletasks()
            
            # Preprocess text
            text = self.preprocess_text(text)
            
            # Gender prediction
            self.status_var.set("Predicting gender...")
            self.progress_var.set(40)
            text_vectorized = self.vectorizer.transform([text])
            gender = self.model.predict(text_vectorized)[0]
            probabilities = self.model.predict_proba(text_vectorized)[0]
            confidence = max(probabilities) * 100

            # Sentiment analysis
            self.status_var.set("Analyzing sentiment...")
            self.progress_var.set(60)
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Determine sentiment label
            if polarity > 0.3:
                sentiment = "Very Positive 😊"
            elif polarity > 0:
                sentiment = "Positive 🙂"
            elif polarity < -0.3:
                sentiment = "Very Negative 😠"
            elif polarity < 0:
                sentiment = "Negative 😔"
            else:
                sentiment = "Neutral 😐"

            # Determine mood
            self.status_var.set("Analyzing mood...")
            self.progress_var.set(80)
            mood = self.analyze_mood(text, polarity, subjectivity)

            # Update GUI with emojis
            self.gender_var.set(f"{gender.title()} {'👨' if gender.lower() == 'male' else '👩'}")
            self.confidence_var.set(f"{confidence:.2f}% {'✅' if confidence > 80 else '⚠️'}")
            self.sentiment_var.set(sentiment)
            self.polarity_var.set(f"{polarity:.2f} (Subjectivity: {subjectivity:.2f})")
            self.mood_var.set(mood)

            # Add to history with emojis
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_entry = f"📅 [{timestamp}]\n"
            history_entry += f"📝 Text: {text}\n"
            history_entry += f"👤 Gender: {gender.title()} ({confidence:.2f}%)\n"
            history_entry += f"😊 Sentiment: {sentiment}\n"
            history_entry += f"🎭 Mood: {mood}\n"
            history_entry += "─" * 50 + "\n"
            
            self.history_text.insert(tk.END, history_entry)
            self.history_text.see(tk.END)
            
            # Complete progress
            self.status_var.set("Analysis complete! ✅")
            self.progress_var.set(100)
            self.root.after(2000, lambda: self.status_var.set(""))
            self.root.after(1000, lambda: self.progress_var.set(0))

        except Exception as e:
            self.status_var.set("Analysis failed! ❌")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.progress_var.set(0)

    def clear_all(self):
        """Clear all input and results."""
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", "Type or paste your text here...")
        self.text_input.configure(fg=self.colors['text_secondary'])
        self.gender_var.set("")
        self.confidence_var.set("")
        self.sentiment_var.set("")
        self.polarity_var.set("")
        self.mood_var.set("")
        self.progress_var.set(0)
        self.status_var.set("")

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
                self.status_var.set("History saved successfully! ✅")
                self.root.after(2000, lambda: self.status_var.set(""))
        except Exception as e:
            self.status_var.set("Failed to save history! ❌")
            messagebox.showerror("Error", f"Failed to save history: {str(e)}")

    def add_tooltip(self, widget, text):
        def show_tooltip(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            label = ttk.Label(self.tooltip, 
                            text=text, 
                            justify='left',
                            background=self.colors['surface'],
                            foreground=self.colors['text'],
                            font=('Segoe UI', 9),
                            relief='solid',
                            borderwidth=1,
                            padding=5)
            label.pack()
        
        def hide_tooltip(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
        
        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)

def main():
    root = tk.Tk()
    app = GenderAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()