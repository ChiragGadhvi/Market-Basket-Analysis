from flask import Flask, render_template, request
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    min_support = float(request.form['min_support'])
    data = pd.read_csv('transactions.csv')  # Replace with your dataset path
    
    # Perform market basket analysis
    one_hot_encoded = pd.get_dummies(data)
    frequent_itemsets = apriori(one_hot_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
    
    return render_template('results.html', rules=rules)

if __name__ == '__main__':
    app.run(debug=True)
