from tkinter import *
from tkinter import ttk
import requests
import urllib.parse
import speech_recognition as sr  # 🎤

# Language codes
LANGUAGES = {
    'english': 'en',
    'hindi': 'hi',
    'french': 'fr',
    'spanish': 'es',
    'german': 'de',
    'japanese': 'ja',
    'chinese': 'zh',
    'arabic': 'ar',
    'russian': 'ru'
}
lang_names = list(LANGUAGES.keys())

# GUI window setup
root = Tk()
root.geometry('950x550')
root.title("Language Translator ✨")
root.configure(bg="#f0f2f5")

Label(root, text="🌐 Language Translator", font=("Helvetica", 24, "bold"), bg="#f0f2f5", fg="#333").pack(pady=10)

# Frames for input/output
input_frame = Frame(root, bg="white", bd=2, relief=GROOVE)
input_frame.place(x=50, y=70, width=400, height=250)

output_frame = Frame(root, bg="white", bd=2, relief=GROOVE)
output_frame.place(x=500, y=70, width=400, height=250)

# Labels
Label(input_frame, text="Enter Text", font=("Helvetica", 12, "bold"), bg="white", fg="black").pack(anchor="w", padx=10, pady=5)
Label(output_frame, text="Translated Text", font=("Helvetica", 12, "bold"), bg="white", fg="black").pack(anchor="w", padx=10, pady=5)

# Text boxes
Input_text = Text(input_frame, font=("Helvetica", 11), wrap=WORD, bd=0, padx=10, pady=5)
Input_text.pack(fill=BOTH, expand=True)

Output_text = Text(output_frame, font=("Helvetica", 11), wrap=WORD, bd=0, padx=10, pady=5)
Output_text.pack(fill=BOTH, expand=True)

# Language dropdowns
src_lang = ttk.Combobox(root, values=lang_names, width=25, font=("Helvetica", 10))
src_lang.place(x=70, y=340)
src_lang.set("choose input language")

dest_lang = ttk.Combobox(root, values=lang_names, width=25, font=("Helvetica", 10))
dest_lang.place(x=520, y=340)
dest_lang.set("choose output language")

# Translate function
def Translate():
    try:
        src = LANGUAGES.get(src_lang.get().lower())
        dest = LANGUAGES.get(dest_lang.get().lower())
        text = Input_text.get(1.0, END).strip()

        if not src or not dest or not text:
            Output_text.delete(1.0, END)
            Output_text.insert(END, "Please select both languages and enter text.")
            return

        encoded_text = urllib.parse.quote(text)
        url = f"https://api.mymemory.translated.net/get?q={encoded_text}&langpair={src}|{dest}"

        response = requests.get(url)
        result = response.json()

        if "responseData" in result and result['responseData']['translatedText']:
            translated_text = result['responseData']['translatedText']
        else:
            translated_text = "Translation failed. Try again."

        Output_text.delete(1.0, END)
        Output_text.insert(END, translated_text)

    except Exception as e:
        Output_text.delete(1.0, END)
        Output_text.insert(END, f"Error: {str(e)}")

# Clear function
def Clear():
    Input_text.delete(1.0, END)
    Output_text.delete(1.0, END)
    src_lang.set("choose input language")
    dest_lang.set("choose output language")

# 🎤 Voice input function
def VoiceInput():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    try:
        Output_text.delete(1.0, END)
        Output_text.insert(END, "Listening... Please speak.")
        root.update()

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)

        text = recognizer.recognize_google(audio)
        Input_text.delete(1.0, END)
        Input_text.insert(END, text)
        Output_text.delete(1.0, END)
        Output_text.insert(END, "Voice captured. Now press 'Translate'.")

    except sr.UnknownValueError:
        Output_text.delete(1.0, END)
        Output_text.insert(END, "Sorry, could not understand your voice.")
    except sr.RequestError:
        Output_text.delete(1.0, END)
        Output_text.insert(END, "API error. Check your internet.")
    except Exception as e:
        Output_text.delete(1.0, END)
        Output_text.insert(END, f"Voice Input Error: {str(e)}")

# Buttons
translate_btn = Button(root, text="Translate", command=Translate, font=("Helvetica", 12, "bold"),
                       bg="#4a90e2", fg="white", padx=10, pady=5, relief=FLAT)
translate_btn.place(x=420, y=400)

clear_btn = Button(root, text="Clear", command=Clear, font=("Helvetica", 12, "bold"),
                   bg="#f44336", fg="white", padx=10, pady=5, relief=FLAT)
clear_btn.place(x=420, y=440)

voice_btn = Button(root, text="🎙️ Voice Input", command=VoiceInput, font=("Helvetica", 12, "bold"),
                   bg="#28a745", fg="white", padx=10, pady=5, relief=FLAT)
voice_btn.place(x=420, y=480)

root.mainloop()
