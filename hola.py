from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__)

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')

    # Validar la longitud del mensaje
    if len(user_message) > 50:
        return jsonify({'error': 'Message exceeds 50 characters limit.'}), 400

    history = [
        {"role": "system", "content": "You are an intelligent assistant."},
        {"role": "user", "content": user_message}
    ]

    completion = client.chat.completions.create(
        model="Kukedlc/SpanishChat-7b-GGUF",
        messages=history,
        temperature=0.8,
        max_tokens=50,
    )

    assistant_message = completion.choices[0].message.content
    return jsonify({'response': assistant_message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
