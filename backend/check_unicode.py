import unicodedata
import codecs

def check_string(name, text, file):
    file.write(f"--- {name} ---\n")
    for char in text:
        cp = ord(char)
        file.write(f"Char: {repr(char)}, CodePoint: U+{cp:04X}, Name: {unicodedata.name(char, 'UNKNOWN')}\n")

with codecs.open("unicode_results.md", "w", "utf-8") as f:
    check_string("Sanskrit", "संस्कृतम्", f)
    check_string("Assamese", "অসমীয়া", f)
    check_string("Urdu", "اردو", f)
