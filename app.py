# support api
from datetime import datetime, timedelta, date
from boto3.dynamodb.conditions import Key
from joblib import Parallel, delayed
from dotenv import load_dotenv
import yfinance as yf
import boto3
import time
import json
import re 
import os

# Updated imports
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from redis.exceptions import ConnectionError as RedisConnectionError
from flask import Flask, jsonify, request, Response
from requests.exceptions import RequestException
from flask_cors import CORS
from redis import Redis
import pandas as pd
import numpy as np
import openai
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"methods": ["GET", "POST", "PUT", "DELETE"]}})

load_dotenv()

openai.organization = os.getenv("OPENAI_API_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")

# MEMORY_SIZE = 10
# memory = []
stock_data_cache = {} 

s3 = boto3.client('s3')
BUCKET_NAME = os.getenv("BUCKET_NAME")
CHAT_HISTORY_PREFIX = 'chat_history/'

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
table = dynamodb.Table(os.getenv("DYNAMODB_TABLE_NAME"))

# Updated Redis setup
try:
    r = Redis(host='localhost', port=6379, db=0)
except RedisConnectionError:
    r = None

@app.route('/')
def run_test():
    return 'live'

@app.route('/be')
def run_test():
    return 'be-live'

# # Updated function get_user_profile()
# def get_user_profile(username):
#     try:
#         response = table.get_item(
#             Key={
#                 'UserID': username
#             }
#         )
#     except NoCredentialsError or PartialCredentialsError:
#         return None

#     user_profile = response.get('Item')
    
#     if user_profile is None:
#         return None

#     user_profile_df = pd.DataFrame([user_profile])

#     list_columns = ['CurrentPortfolio', 'PortfolioHistory', 'Recommendations', 'AllocationPercentages']
#     for column in list_columns:
#         user_profile_df[column] = user_profile_df[column].apply(json.loads)
    
#     return user_profile_df

# def update_user_profile(user_profile):
#     """
#     This function updates a user's profile in the DynamoDB database.
#     """
#     table.put_item(
#         Item=user_profile
#     )

# @app.route('/api/userPortfolio/<username>', methods=['GET'])
# def api_user_portfolio(username):
#     if username is None or username.lower() == 'undefined':
#         return jsonify({'message': 'Invalid username supplied'}), 400

#     user_profile_df = get_user_profile(username)

#     if user_profile_df is None:
#         return jsonify({'message': 'User not found'}), 404

#     user_profile_dict = user_profile_df.to_dict(orient='records')[0]
#     return jsonify(user_profile_dict), 200
    
# @app.route('/api/userPortfolio/<username>', methods=['PUT'])
# def api_update_user_portfolio(username):
#     if username is None or username.lower() == 'undefined':
#         return jsonify({'message': 'Invalid username supplied'}), 400

#     user_profile = request.json
#     user_profile['UserID'] = username

#     list_columns = ['CurrentPortfolio', 'PortfolioHistory', 'Recommendations', 'AllocationPercentages']
#     for column in list_columns:
#         user_profile[column] = json.dumps(user_profile[column])

#     update_user_profile(user_profile)
#     return jsonify({'message': 'User profile updated successfully'}), 200

# @app.route('/api/feedback', methods=['POST'])
# def api_feedback():
#     data = request.get_json()
#     print("Received data:", data)

#     message_id = data.get('message_id')
#     feedback = data.get('feedback')

#     if message_id and feedback:
#         feedback_data = {
#             'message_id': message_id,
#             'feedback': feedback,
#             'username': data.get('username')
#         }

#         table.put_item(Item=feedback_data)
#         print('success', feedback_data)
#         return jsonify({'status': 'success'})
#     else:
#         return jsonify({'error': 'Invalid input'}), 400

# def save_chat_history(chat_history):
#     today = datetime.utcnow().strftime("%Y-%m-%d")
#     chat_history_key = f"{CHAT_HISTORY_PREFIX}chat_history_{today}.json"
    
#     print('save chat_history::',chat_history)

#     s3.put_object(
#         Bucket=BUCKET_NAME,
#         Key=chat_history_key,
#         Body=json.dumps(chat_history)
#     )

# # Updated function load_chat_history()
# def load_chat_history():
#     today = datetime.utcnow().strftime("%Y-%m-%d")
#     chat_history_key = f"{CHAT_HISTORY_PREFIX}chat_history_{today}.json"
    
#     try:
#         response = s3.get_object(Bucket=BUCKET_NAME, Key=chat_history_key)
#     except NoCredentialsError or PartialCredentialsError:
#         return []
    
#     chat_history = json.loads(response['Body'].read())
#     # Filter out non-dictionary entries
#     chat_history = [chat for chat in chat_history if isinstance(chat, dict)]

#     return chat_history[:7]

# def simulate_single_portfolio(mean_returns, cov_matrix, risk_free_rate):
#     num_assets = len(mean_returns)
#     weights = np.random.random(num_assets)
#     weights /= np.sum(weights)
#     returns = np.dot(weights, mean_returns)
#     volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
#     sharpe_ratio = (returns - risk_free_rate) / volatility
#     return returns, volatility, sharpe_ratio, weights

# def get_key_allocations(results, weights_record, stock_data):
#     max_sharpe_idx = np.argmax(results[2])
#     sdp, rp = results[1, max_sharpe_idx], results[0, max_sharpe_idx]
#     max_sharpe_allocation = pd.DataFrame(weights_record[max_sharpe_idx], index=stock_data.columns, columns=['allocation'])
#     max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
#     max_sharpe_allocation = max_sharpe_allocation.T

#     min_vol_idx = np.argmin(results[1])
#     sdp_min, rp_min = results[1, min_vol_idx], results[0, min_vol_idx]
#     min_vol_allocation = pd.DataFrame(weights_record[min_vol_idx], index=stock_data.columns, columns=['allocation'])
#     min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
#     min_vol_allocation = min_vol_allocation.T

#     return max_sharpe_allocation, min_vol_allocation

# def simulate_portfolios_parallel(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
#     with Parallel(n_jobs=-1, prefer="threads") as parallel:
#         results_and_weights = parallel(
#             delayed(simulate_single_portfolio)(mean_returns, cov_matrix, risk_free_rate)
#             for _ in range(num_portfolios)
#         )

#     results = np.array([item[0:3] for item in results_and_weights]).T
#     weights_record = [item[3] for item in results_and_weights]

#     return results, weights_record

# PROFILE_PREFIX = 'user_profile/'

# def save_user_portfolio_to_dynamodb(portfolio, temp_portfolio, username):
#     """
#     Save a user's portfolio and temporary portfolio to a DynamoDB table.
#     """

#     print("portfolio + username::", portfolio, username)

#     table.put_item(
#         Item={
#             'username': username,
#             'portfolio': json.dumps(portfolio),
#             'temp_portfolio': json.dumps(temp_portfolio)
#         }
#     )

# def load_user_profile_from_dynamodb(username):
#     """
#     Load a user's portfolio from DynamoDB. 
#     If the user has a saved portfolio, load it as the temporary portfolio.
#     """
#     try:
#         response = table.get_item(Key={'username': username})
#     except ClientError as e:
#         print(e.response['Error']['Message'])
#     else:
#         item = response.get('Item')
#         if item is None:
#             return None

#         portfolio = json.loads(item.get('portfolio', '{}'))
#         temp_portfolio = json.loads(item.get('temp_portfolio', '{}'))

#         return {
#             'portfolio': portfolio,
#             'temp_portfolio': temp_portfolio or portfolio  # Use the saved portfolio as the default temp portfolio
#         }

# @app.route('/api/efficient_frontier', methods=['POST'])
# def efficient_frontier():
#     data = request.json
#     username = data.get('username', 'tsm')
#     prompt = data.get('prompt', '').lower()

#     user_profile = load_user_profile_from_dynamodb(username)
    
#     # Parse tickers from the prompt
#     prompt_tickers = prompt.split('$')[1].split()    
#     prompt_tickers = [ticker.upper() for ticker in prompt_tickers]

#     print("prompt_tickers::", prompt_tickers)
#     # Create an empty dictionary for the loaded user portfolio
#     loaded_user_portfolio = {}

#     for ticker in prompt_tickers:
#         if user_profile is None or ticker not in user_profile:
#             loaded_user_portfolio[ticker] = {"allocation": 0}  # default allocation

#     tickers = list(loaded_user_portfolio.keys())
#     allocations = [v["allocation"] for v in loaded_user_portfolio.values()]

#     stock_data_key = tuple(sorted(tickers))
#     if stock_data_key not in stock_data_cache:
#         stock_data_cache[stock_data_key] = yf.download(tickers, start='2022-06-01', end='2023-06-01')['Adj Close']

#     stock_data = stock_data_cache[stock_data_key]
#     daily_returns = stock_data.pct_change()
#     daily_returns = daily_returns.dropna()

#     mean_returns = daily_returns.mean()
#     cov_matrix = daily_returns.cov()

#     risk_free_rate = 0.045
#     num_portfolios = 10000
#     results, weights_record = simulate_portfolios_parallel(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

#     max_sharpe_allocation, min_vol_allocation = get_key_allocations(results, weights_record, stock_data)

#     max_sharpe_index = results[2, :].argmax()
#     min_vol_index = results[1, :].argmin()

#     print("ef_max_sharpe_allocation:: ", max_sharpe_allocation.to_dict())
#     print("ef_min_volatility_allocation:: ", min_vol_allocation.to_dict())

#     user_risk_tolerance = "high"

#     if user_risk_tolerance == "high":
#         portfolio_allocation = max_sharpe_allocation
#     elif user_risk_tolerance == "low" or user_risk_tolerance == "medium": 
#         portfolio_allocation = min_vol_allocation
#     else: 
#         portfolio_allocation = min_vol_allocation

#     # Parse the prompt for 'save' command 
#     if 'save' in prompt:
#         save_user_portfolio_to_dynamodb(portfolio_allocation.to_dict(), username)
#         return jsonify({'message': 'Portfolio saved successfully'}), 200

#     return jsonify({
#         'tickers': tickers,
#         'allocations': allocations,
#         'portfolio_allocation': portfolio_allocation.to_dict(),    
#         'results': results.tolist(),
#         'weights_record': [w.tolist() for w in weights_record],
#         'max_sharpe_index': int(max_sharpe_index),
#         'min_vol_index': int(min_vol_index),
#     })

# Updated api_combined_summary()
@app.route('/api/combined_summary', methods=['POST'])
def api_combined_summary():
    data = request.get_json()
    query = data.get('query')
    num_results = data.get('num_results', 10)
    username = data.get('username', 'tsm')

    try:
        user_profile = load_user_profile_from_dynamodb(username)
    except NoCredentialsError or PartialCredentialsError:
        return jsonify({'error': 'Unable to load user profile'}), 500
        
    user_prompt = {
        'timestamp': time.time(),
        'sender': username,
        'message': query
    }
    
    print('cs_user_prompt::',user_prompt)
    print('cs_user_profile::',user_profile)

    if query != "":
        gpt_prompt = f"Remember to respond in a tone similar to the input prompt in 250 characters or less. You can provide more details on request: {user_profile}'{query}'\n\n"

        # portfolio = '\n'.join([f"{ticker}: {details['allocation']}%" for ticker, details in user_profile.items()])

        # gpt_prompt += f"\n Portfolio {portfolio}\n"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI portfolio manager. You are equipped with profit growth algos that maximize returns while minimizing risk. respond in this I have these tickers $ NVDA MSFT now add these $ AMZN the response would include at the end $ NVDA MSFT AMZN",
                },
                {"role": "user", "content": gpt_prompt},
            ],
        )

        gpt_response = response.choices[0].message['content'].strip()
        return jsonify({'combined_summary': gpt_response, 'user_profile': user_profile})

    else:  # Query is empty, use Google Search instead.
        combined_summary = "Can you provide more details on what you're looking to do?" #get_combined_summary(query, num_results)
        return jsonify({'combined_summary': combined_summary, 'user_profile': user_profile})
    
if __name__ == "__main__":
    # app.run(port=5000)
    app.run(host="0.0.0.0", port=8080)
