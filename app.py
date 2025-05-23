
from flask import Flask, request, render_template
from model import generate_response

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    response = ""
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        response = generate_response(prompt)
    return render_template('index.html', response=response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
