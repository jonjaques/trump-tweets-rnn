from flask import Flask, render_template, jsonify, request
from textgenrnn import textgenrnn

app = Flask(__name__)
tgen = textgenrnn(
  name='trump_char1',
  weights_path='trump_char1_weights.hdf5',
  vocab_path='trump_char1_vocab.json',
  config_path='trump_char1_config.json'
)

print(tgen.generate())

@app.route('/')
def index():
  return render_template('index.liquid')

@app.route('/api/generate')
def generate():
  num = request.args.get('num')
  temp = request.args.get('temp')

  if num is None:
    num = 1

  if temp is None:
    temp = 0.35

  return jsonify(tgen.generate(
    n=int(num), 
    temperature=float(temp), 
    return_as_list=True
  ))