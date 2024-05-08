from flask import Flask, request

app = Flask(__name__)

@app.route('/api/nlp', methods=['POST'])
def index():
    json = request.get_json()
    print(json['tell1'])
    return 'tell1'

if __name__ == '__main__':
    app.run(debug=False)