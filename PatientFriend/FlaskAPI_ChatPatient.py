from flask import Flask, request, jsonify
from ChatPatient import ChatPatient

app = Flask(__name__)
chat_patient = ChatPatient()

@app.route('/api/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    report_text = data.get('text')
    if not report_text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        summary = chat_patient.get_friendly_text(report_text)
        return jsonify({'summary': summary}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    text = data.get('text')
    language = data.get('language', 'hindi')  # Default language is Hindi
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        translated_text = chat_patient.translate_text(text, language=language)
        return jsonify({'translated_text': translated_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
