from flask import Flask, render_template, request
from prediction_model import name_get

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/process_form', methods=['POST'])
def process_form():
    stock = request.form.get('stock')
    start = request.form.get('start')
    end = request.form.get('end')
    

    if not (stock and start and end):
        return "Error: All form fields must be filled."
    
    name_get(stock, start, end)
    # state = 'True'

    return render_template('predic.html',stock_name=stock,
                           start_date = start,
                           end_date = end)

if __name__ == '__main__':
    app.run(debug=True)