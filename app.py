from flask import Flask, request, jsonify, render_template
import torch
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load model and tokenizer
languages = {
        "English": "en",
        "German": "de",
        "French": "fr",
    }
language_pairs = [(languages[lang1], languages[lang2]) for lang1 in languages for lang2 in languages if lang1 != lang2]

def get_model_name(source_lang, target_lang):
        return f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

failed_downloads = []

for source_lang, target_lang in language_pairs:
        model_name = get_model_name(source_lang, target_lang)
        try:
            print(f"Attempting to download model for {source_lang}-{target_lang}")
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            print(f"Successfully downloaded {model_name}")
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
            failed_downloads.append(model_name)

print("Model downloading phase completed.")

@app.route('/')
def home():
    return render_template('index.html')  # Assuming your HTML file is named index.html

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content to return

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json

    text_to_translate = data['text']
    source_lang = data['src_lang']
    target_lang = data['tgt_lang']

    # Prepare the text for translation
    model_name = get_model_name(source_lang, target_lang)
    tokenizer = MarianTokenizer.from_pretrained(model_name, use_cache=True)
    model = MarianMTModel.from_pretrained(model_name, use_cache=True)
    
    # Encode the text and generate translation
    tokenized_text = tokenizer.prepare_seq2seq_batch([text_to_translate], return_tensors='pt')
    translation = model.generate(**tokenized_text)
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)

    return jsonify({'translated_text': translated_text})

if __name__ == '__main__':
    app.run(debug=True, port=5500)
