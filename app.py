from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load model and tokenizer
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

@app.route('/')
def home():
    return render_template('index.html')  # Assuming your HTML file is named index.html

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content to return

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data['text']
    src_lang = data['src_lang']
    tgt_lang = data['tgt_lang']

    # Prepare the text for translation
    input_text = f"translate {src_lang} to {tgt_lang}: {text}"
    
    # Encode the text and generate translation
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'translated_text': translated_text})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
