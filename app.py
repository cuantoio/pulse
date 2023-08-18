# support api
from datetime import datetime, timedelta, date
from boto3.dynamodb.conditions import Key, Attr
from joblib import Parallel, delayed
from dotenv import load_dotenv
from scipy.stats import norm
from prophet import Prophet
import yfinance as yf
import traceback
import boto3
import time
import json
import ast
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
import stripe
import os

app = Flask(__name__)
CORS(app)

load_dotenv()

openai.organization = "org-bHNvimGOVpNGPOLseRrHQTB4"
openai.api_key = os.getenv("OPENAI_API_KEY")
stripe.api_key = os.getenv("STRIPE_API_KEY")

# MEMORY_SIZE = 10
# memory = []
stock_data_cache = {} 

s3 = boto3.client('s3')
BUCKET_NAME = os.getenv("BUCKET_NAME")
CHAT_HISTORY_PREFIX = 'chat_history/'

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
table = dynamodb.Table(os.getenv("DYNAMODB_TABLE_INPUT_PROMPTS"))
table_portfolio = dynamodb.Table(os.getenv("DYNAMODB_TABLE_PORTFOLIO"))
table_user_profiles = dynamodb.Table(os.getenv("DYNAMODB_TABLE_USER_PROFILE"))
table_chat_history = dynamodb.Table(os.getenv("DYNAMODB_TABLE_CHAT_HISTORY"))
table_payments = dynamodb.Table(os.getenv("DYNAMODB_TABLE_PAYMENTS")) 

# Updated Redis setup
try:
    r = Redis(host='localhost', port=6379, db=0)
except RedisConnectionError:
    r = None

@app.route('/')
def run_test():
    print("get user")
    return 'live'

@app.route('/eb')
def run_eb():
    return 'eb-live v5.0.1'

# Updated function get_user_profile()
def get_user_profile(username):
    try:
        response = table.get_item(
            Key={
                'UserID': username
            }        
        )        
    except NoCredentialsError or PartialCredentialsError:
        return None

    user_profile = response.get('Item')
    
    if user_profile is None:
        return None

    user_profile_df = pd.DataFrame([user_profile])

    list_columns = ['CurrentPortfolio', 'PortfolioHistory', 'Recommendations', 'AllocationPercentages']
    for column in list_columns:
        user_profile_df[column] = user_profile_df[column].apply(json.loads)
    
    return user_profile_df

def update_user_profile(user_profile):
    """
    This function updates a user's profile in the DynamoDB database.
    """
    table.put_item(
        Item=user_profile
    )
    print("update called")

@app.route('/api/userPortfolio/<username>', methods=['GET'])
def api_user_portfolio(username):
    if username is None or username.lower() == 'undefined':
        return jsonify({'message': 'Invalid username supplied'}), 400

    # user_profile_df = get_user_profile(username)
    # user_portfolio = load_user_portfolio_from_dynamodb(username)

    if user_portfolio is None:
        return jsonify({'message': 'User not found'}), 404

    user_portfolio_dict = user_portfolio.to_dict(orient='records')[0]
    return jsonify(user_portfolio_dict), 200

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    data = request.get_json()
    print("Received data:", data)

    message_id = data.get('message_id')
    feedback = data.get('feedback')

    if message_id and feedback:
        feedback_data = {
            'message_id': message_id,
            'feedback': feedback,
            'username': data.get('username')
        }

        table.put_item(Item=feedback_data)
        
        return jsonify({'status': 'success'})
    else:
        return jsonify({'error': 'Invalid input'}), 400

def save_chat_history(chat_history):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    chat_history_key = f"{CHAT_HISTORY_PREFIX}chat_history_{today}.json"
    
    print('save chat_history::',chat_history)

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=chat_history_key,
        Body=json.dumps(chat_history)
    )

# Updated function load_chat_history()
def load_chat_history():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    chat_history_key = f"{CHAT_HISTORY_PREFIX}chat_history_{today}.json"
    
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=chat_history_key)
    except NoCredentialsError or PartialCredentialsError:
        return []
    
    chat_history = json.loads(response['Body'].read())
    # Filter out non-dictionary entries
    chat_history = [chat for chat in chat_history if isinstance(chat, dict)]

    return chat_history[:7]

def value_at_risk(portfolio_returns, confidence_level=0.05):
    """
    Calculate Value at Risk (VaR) of a portfolio at a specified confidence level
    """
    return np.percentile(portfolio_returns, 100 * (1 - confidence_level))

def conditional_value_at_risk(portfolio_returns, confidence_level=0.05):
    """
    Calculate Conditional Value at Risk (CVaR) of a portfolio at a specified confidence level
    """
    var = value_at_risk(portfolio_returns, confidence_level)
    return portfolio_returns[portfolio_returns <= var].mean()

def simulate_single_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (returns - risk_free_rate) / volatility
    var = value_at_risk(returns)
    cvar = conditional_value_at_risk(returns)

    return returns, volatility, sharpe_ratio, weights, var, cvar

def get_key_allocations(results, weights_record, stock_data):
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[1, max_sharpe_idx], results[0, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights_record[max_sharpe_idx], index=stock_data.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [i * 100 for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[1])
    sdp_min, rp_min = results[1, min_vol_idx], results[0, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights_record[min_vol_idx], index=stock_data.columns, columns=['allocation'])
    min_vol_allocation.allocation = [i * 100 for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    return max_sharpe_allocation, min_vol_allocation

def simulate_portfolios_parallel(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    with Parallel(n_jobs=-1, prefer="threads") as parallel:
        results_and_weights = parallel(
            delayed(simulate_single_portfolio)(mean_returns, cov_matrix, risk_free_rate)
            for _ in range(num_portfolios)
        )

    results = np.array([item[0:6] for item in results_and_weights]).T
    weights_record = [item[3] for item in results_and_weights]

    # Now we calculate risk and return scores after all portfolios have been simulated.
    returns = results[0, :]
    volatilities = results[1, :]
    epsilon = 1e-10  # A small value
    risk_scores = (volatilities - np.min(volatilities)) / (np.max(volatilities) - np.min(volatilities) + epsilon) * 100
    return_scores = (returns - np.min(returns)) / (np.max(returns) - np.min(returns) + epsilon) * 100

    return results, weights_record, risk_scores, return_scores

def format_data(data):
    return ' '.join(f'\n{i}. ${ticker} | {info["allocation"]}%' for i, (ticker, info) in enumerate(data.items(), start=1))

def calculate_score(total_allocation, tracking_error, active_share):
    # Define the weights for each metric
    total_allocation_weight = 0.33
    tracking_error_weight = 0.33
    active_share_weight = 0.33

    # Normalizing the metrics
    normalized_total_allocation = total_allocation / 100
    normalized_tracking_error = tracking_error / 100
    normalized_active_share = active_share / 100

    # Calculating the combined score
    score = (total_allocation_weight * normalized_total_allocation +
             tracking_error_weight * normalized_tracking_error +
             active_share_weight * normalized_active_share)

    return score * 100  # converting the score to percentage

@app.route('/api/efficient_frontier', methods=['POST'])
def efficient_frontier():
    data = request.json
    username = data.get('username', 'noname')
    prompt = data.get('query', '').lower()

    print(f"Data received: {data}")

    prompt_tickers = data.get('tickers', [])

    # You mentioned you commented out the database function.
    # user_profile = load_user_portfolio_from_dynamodb(username)
    user_profile = {}

    print(f"Prompt tickers after parsing and conversion: {prompt_tickers}")

    # Optimized creation of loaded_user_portfolio using dictionary comprehension
    loaded_user_portfolio = {ticker: {"allocation": 0} for ticker in prompt_tickers if not user_profile or ticker not in user_profile}

    print(f"Loaded user portfolio: {loaded_user_portfolio}")

    tickers = list(loaded_user_portfolio.keys())
    print(f"Final list of tickers: {tickers}")

    allocations = [v["allocation"] for v in loaded_user_portfolio.values()]

    # Use today's date as the end date and one year ago as the start date
    today = date.today()
    start_date = today - timedelta(days=365)
    end_date = today

    bad_tickers = []
    stock_data_key = tuple(sorted(tickers))
    
    # If data isn't in the cache, download it
    if stock_data_key not in stock_data_cache:
        try:
            stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

            if stock_data.empty:
                bad_tickers.extend(tickers)
            else:
                # Identify missing columns (bad tickers)
                fetched_tickers = stock_data.columns.tolist()
                for ticker in tickers:
                    if ticker not in fetched_tickers:
                        bad_tickers.append(ticker)
                stock_data_cache[stock_data_key] = stock_data
        except Exception as e:
            print(f"Failed to download data. Error: {str(e)}")
            return jsonify({'error': f"Failed to download data. Error: {str(e)}"}), 500

    else:
        stock_data = stock_data_cache[stock_data_key]
    
    print(f"Successfully downloaded tickers: {[ticker for ticker in tickers if ticker not in bad_tickers]}")
    if bad_tickers:
        print(f"Bad tickers list: {bad_tickers}")

    stock_data = stock_data_cache[stock_data_key]
    daily_returns = stock_data.pct_change()
    daily_returns = daily_returns.dropna()

    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    n_sims = 1000
    risk_free_rate = 0.045
    num_portfolios = n_sims

    # results, weights_record = simulate_portfolios_parallel(num_portfolios, mean_returns, cov_matrix, risk_free_rate)
    results, weights_record, risk_scores, return_scores = simulate_portfolios_parallel(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_index = results[2, :].argmax()
    min_vol_index = results[1, :].argmin()

    if ("low" in prompt or "medium" in prompt) and "risk" in prompt:
        print("low risk")
        user_risk_tolerance = "low"
    else:
        print("high risk")
        user_risk_tolerance = "high"

    # Calculate the portfolio's risk and return scores
    risk_score = risk_scores[max_sharpe_index] if user_risk_tolerance == "high" else risk_scores[min_vol_index]
    return_score = return_scores[max_sharpe_index] if user_risk_tolerance == "high" else return_scores[min_vol_index]

    max_sharpe_allocation, min_vol_allocation = get_key_allocations(results, weights_record, stock_data)

    print(f"Max Sharpe Allocation: {max_sharpe_allocation.to_dict()}")
    print(f"Min Volatility Allocation: {min_vol_allocation.to_dict()}")
    
    if user_risk_tolerance == "low" or user_risk_tolerance == "medium": 
        portfolio_allocation = min_vol_allocation
    else: 
        portfolio_allocation = max_sharpe_allocation
    
    print(f"print portfolio_allocation: {portfolio_allocation.to_dict()}")
    blended_portfolio = fetch_data(portfolio_allocation.to_dict(), '')
    print(f"print blended_portfolio:: {blended_portfolio}")

    # save portfolio to chat history v1.06
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    portfolio_summary = format_data(portfolio_allocation)

    chat = {
        "query": prompt_tickers,
        "trianglai_response": portfolio_summary
    }
    save_chat_history(username, timestamp, chat)

    # save portfolio
    if 'save' in prompt:
        save_user_portfolio_to_dynamodb(portfolio_allocation.to_dict(), username)
        # return jsonify({'message': 'Portfolio saved successfully'}), 200

    print('risk_score', risk_score,'return_score', return_score)

    return jsonify({
        'tickers': tickers,
        'allocations': allocations,
        'portfolio_allocation': portfolio_allocation.to_dict(),    
        'results': [[r.tolist() if isinstance(r, np.ndarray) else r.item() for r in res] for res in results.tolist()],
        'weights_record': [[w.item() for w in weights] for weights in weights_record],
        'max_sharpe_index': int(max_sharpe_index),
        'min_vol_index': int(min_vol_index),
        'risk_score': risk_score.item() if np.isscalar(risk_score) else risk_score,
        'return_score': return_score.item() if np.isscalar(return_score) else return_score
    })

def save_prompt(prompt):
    table.put_item(
        Item={
            'username': prompt['username'],
            'timestamp': prompt['timestamp'],
            'query': prompt['query']
        }
    )

def save_chat_history(username, timestamp, chat):
    table_chat_history.put_item(
        Item={
            'username': username,
            'timestamp': timestamp,
            'chat': chat
        }
    )

def load_chat_history(username):
    response = table_chat_history.query(
        KeyConditionExpression=Key('username').eq(username),
        ScanIndexForward=False,  # to retrieve items in descending order
        Limit=7  # limit to last conversation
    )
    
    items = response['Items']
    
    if items:
        # Return the 'chat' dictionary of the last conversation
        return items[0]['chat']

    return None  # return None if no conversation found

@app.route('/api/combined_summary', methods=['POST'])
def api_combined_summary():
    data = request.json
    query = data.get('query')
    num_results = data.get('num_results', 10)
    username = data.get('username', 'tsm')

    # received prompt data
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # user_portfolio = load_user_portfolio_from_dynamodb(username)

    gpt_prompt = f"""given this query: {query}"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You're an AI that chats in short, friendly and conversational manner. You provide portfolio updates in a format that anyone can understand, never in JSON format. Unless the user's query involves analyzing the optimized portfolio, you assist with ticker lists, strategey, recommendations, portfolio optimization, personalized financial planning, and real-time market analysis. provided tickers must be in this format: [$TICKER1 $TICKER2]",
                # "content": "You are an AI Optimizer. You help financial professionals balance client portfolios, recommend strategy optimization, strategy brainstorming and more. Try to keep responses under 150 characters."
            },
            {
                "role": "user", 
                "content": gpt_prompt
            },
        ],
    )

    gpt_response = response.choices[0].message['content'].strip()

    # Save chat history to DynamoDB
    chat = {
        "query": query,
        "trianglai_response": gpt_response
    }
    save_chat_history(username, timestamp, chat)

    return jsonify({'trianglai_response': gpt_response, 'username': username})

# default_portfolio = {
#     'DLR': {'allocation': 18.58134053},
#     'SPG': {'allocation': 5.96591771},
#     'ARE': {'allocation': 5.21638715},
#     'EQR': {'allocation': 7.01589525},
#     'NVDA': {'allocation': 7.32163178},
#     'EQIX': {'allocation': 7.94139182},
#     'JNJ': {'allocation': 2.48767978},
#     'VOO': {'allocation': 3.68012278},
#     'AEP': {'allocation': 1.52550893},
#     'MSFT': {'allocation': 3.86296522},
#     'AVB': {'allocation': 1.10375472},
#     'NEE': {'allocation': 1.76459701},
#     'META': {'allocation': 6.89370643},
#     'XLU': {'allocation': 0.52595852},
#     'GOOGL': {'allocation': 2.48027440},
#     'IAU': {'allocation': 0.77721259},
#     'WELL': {'allocation': 0.37203234},
#     'WY': {'allocation': 0.33976603},
#     'BA': {'allocation': 1.28836035},
#     'D': {'allocation': 0.14704975},
#     'O': {'allocation': 0.10755437},
#     'GLD': {'allocation': 0.20858496},
#     'AGG': {'allocation': 0.03544005},
#     'TLT': {'allocation': 0.59754389},
#     'AMZN': {'allocation': 0.65378954},
#     'RTX': {'allocation': 1.78152360},
#     'AAPL': {'allocation': 3.65843559},
#     'AMD': {'allocation': 3.81236177},
#     'TSLA': {'allocation': 2.72006775},
#     'ETH-USD': {'allocation': 7.12062853},
#     'DOGE-USD': {'allocation': 0.01251686},
#     'CASH': {'allocation': 0.00000000},
# }

# default_portfolio = { 'META': {'allocation': 25}, 'AAPL': {'allocation': 25}, }

# default_portfolio = {
#     'AAPL': {'allocation': 1.00},
#     'AMZN': {'allocation': 12.00},
#     'GM': {'allocation': 26.00},
#     'META': {'allocation': 1.00},
#     'MSFT': {'allocation': 1.00},
#     'NVDA': {'allocation': 11.00},
#     'TSLA': {'allocation': 2.00},
#     'CASH': {'allocation': 1735.16},
# }

# default_portfolio = {
#     'QQQ': {'allocation': 0.12270776},
#     'VYM': {'allocation': 0.07852126},
#     'NVDA': {'allocation': 0.33522165},
#     'VOO': {'allocation': 0.18964405},
#     'AAPL': {'allocation': 0.15792791},
#     'MSFT': {'allocation': 0.11597737},
#     'CASH': {'allocation': -0.00000000},
# }

PROFILE_PREFIX = 'user_profile/'

def save_user_portfolio_to_dynamodb(dynamodb_entry):
    """
    Save a user's portfolio and temporary portfolio to a DynamoDB table.
    """

    print("dynamodb_entry::", dynamodb_entry)

    table_portfolio.put_item(
        Item={
            'UserId': dynamodb_entry["UserId"],
            'Investments': json.dumps(dynamodb_entry["Investments"]),
            'PortfolioName': dynamodb_entry["PortfolioName"],
        }
    )

def load_user_portfolio_from_dynamodb(UserId, PortfolioName):
    """
    Load a user's portfolio from DynamoDB. 
    If the user has a saved portfolio, load it as the temporary portfolio.
    """
    try:
        response = table_portfolio.get_item(Key={'UserId': UserId, 'PortfolioName': PortfolioName})
        print("Load user")
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        item = response.get('Item')
        if item is None:
            return None

        Investments = json.loads(item.get('Investments', '{}'))

        return {
            'UserId': UserId,
            'Investments': Investments,
            'PortfolioName': PortfolioName,
        }

### SCENARIOS - START ###
import yfinance as yf
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
import concurrent.futures

# default_portfolio = {
#     'IJR': {'allocation': 26.2},
#     'EFA': {'allocation': 25.94},
#     'IJH': {'allocation': 23.83},
#     'VNQ': {'allocation': 14.99},
#     'EEM': {'allocation': 4.09},
#     'SPY': {'allocation': 3.82},
#     'IEF': {'allocation': 0.85},
#     'LQD': {'allocation': 0.29},
# }

default_portfolio = {
  'AAPL': {'allocation': 12.5},
  'AMZN': {'allocation': 12.5},
  'BTC-USD': {'allocation': 12.5},
  'GOOGL': {'allocation': 12.5},
  'META': {'allocation': 12.5},
  'MSFT': {'allocation': 12.5},
  'NVDA': {'allocation': 12.5},
  'TSLA': {'allocation': 12.5},
}

@app.route('/api/parsePortfolio', methods=['POST'])
def api_parse_portfolio():
    data = request.json
    username = data.get('username', 'tsm')
    portfolio_name = data.get('portfolioName', 'Test Portfolio')

    gpt_prompt = f"""given uploaded portfolio: {data}"""

    print("parser:: gpt_prompt:", gpt_prompt)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "please parse the portfolio data given the uploaded portfolio. The response must be in a format like this: [{'META': {'allocation': 25}, 'AAPL': {'allocation': 25}, }]",
                },
                {
                    "role": "user", 
                    "content": gpt_prompt
                },
            ],
        )

        gpt_response = response.choices[0].message['content'].strip()
        print("scenarios:: trianglai_response:", gpt_response)
        gpt_response_data = parse_portfolio_data(gpt_response)
        print("scenarios:: trianglai_response_data:", gpt_response_data)

        # Create a dictionary for the DynamoDB entry
        dynamodb_entry = {
            "UserId": username,
            "PortfolioName": portfolio_name,
            "Investments": gpt_response_data
        }

        #save portfolio
        print("new feat::: dynamodb_entry:", dynamodb_entry)
        save_user_portfolio_to_dynamodb(dynamodb_entry)

        # Return the DynamoDB entry as JSON
        return jsonify(dynamodb_entry), 200

    except Exception as e:
        # If there is an error, return a error message and a 500 status
        return jsonify({"error": str(e)}), 500
    
def extract_events_dates(text):
    pattern = r"'event': '([^']+)', 'date': '([^']+)'"
    matches = re.findall(pattern, text)
    dates = {match[1]: match[0] for match in matches}
    return dates
    
def parse_date(date):
    if ' ongoing' in date or ' onwards' in date:
        date = date.split('-')[0].strip()
        if len(date.split(' ')[0]) == 4:  # it is a year
            return datetime.strptime(date + "-01-01", "%Y-%m-%d")
        else:
            return datetime.strptime(date + "-01", "%Y-%m-%d")
    else:
        try:
            return datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            print(f"Date {date} is not in the expected format 'YYYY-MM-DD'. Trying to correct it.")
            if len(date) == 7:  # Check if date is in 'YYYY-MM' format
                date += '-01'
            try:
                return datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                print(f"Failed to correct the date {date}.")
                return None  # Or handle it as appropriate for your use case

# def fetch_data(portfolio, text):
#     events = extract_events_dates(text)
#     event_dates = sorted([parse_date(date) for date in events.keys()])

#     # Handle the case when no events are found
#     if event_dates:
#         start_date = event_dates[0] - relativedelta(months=6)
#         end_date = event_dates[-1] + relativedelta(months=6)
#     else:
#         # Define arbitrary start and end date, or modify as needed
#         start_date = datetime.today() - relativedelta(years=1)
#         end_date = datetime.today()

#     tickers = list(portfolio.keys())
#     data = {}
#     bad_tickers = []
    
#     for ticker in tickers:
#         try:
#             data[ticker] = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')['Adj Close']
#         except Exception as exc:
#             print('%r generated an exception: %s' % (ticker, exc))
#             bad_tickers.append(ticker)

#     data_pct_change = pd.concat(data, axis=1).pct_change() + 1
#     data_pct_change.iloc[0] = 1 
#     data_pct_change_cumprod = data_pct_change.cumprod()
#     data_pct_change_cumprod = (data_pct_change_cumprod - 1)*100

#     blended_portfolio = pd.Series(0, index=data_pct_change_cumprod.index)
#     for ticker in portfolio:
#         if ticker not in bad_tickers:
#             allocation = portfolio[ticker]['allocation']
#             if allocation is not None:
#                 blended_portfolio += data_pct_change_cumprod[ticker].fillna(0) * allocation / 100
#             else:
#                 print(f"Warning: Allocation for {ticker} is None.")

#     blended_portfolio_df = pd.DataFrame(blended_portfolio, columns=['blended_portfolio'])
#     blended_portfolio_df['norm_close'] = blended_portfolio_df['blended_portfolio']
    
#     print(f"Bad tickers: {bad_tickers}")
#     return blended_portfolio_df, events

def fetch_data(portfolio, text):
    # Extract events and dates
    if text == '':
        events = [{'event': '', 'date': '', 'event_sent': 0.0}]
        start_date = datetime.today() - relativedelta(years=1)
        end_date = datetime.today()
    else:
        events = extract_events_dates(text)
        event_dates = sorted([parse_date(date) for date in events.keys()])

        # Determine the start and end dates based on event dates, or set default values
        if event_dates:
            start_date = event_dates[0] - relativedelta(months=6)
            end_date = event_dates[-1] + relativedelta(months=6)
        else:
            start_date = datetime.today() - relativedelta(years=1)
            end_date = datetime.today()

    tickers = list(portfolio.keys())
    data = {}
    bad_tickers = []
    
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')['Adj Close']
        except Exception as exc:
            print('%r generated an exception: %s' % (ticker, exc))
            bad_tickers.append(ticker)

    data_pct_change = pd.concat(data, axis=1).pct_change() + 1
    data_pct_change.iloc[0] = 1 
    data_pct_change_cumprod = data_pct_change.cumprod()
    data_pct_change_cumprod = (data_pct_change_cumprod - 1)*100

    blended_portfolio = pd.Series(0, index=data_pct_change_cumprod.index)
    for ticker in portfolio:
        if ticker not in bad_tickers:
            allocation = portfolio[ticker]['allocation']
            if allocation is not None:
                blended_portfolio += data_pct_change_cumprod[ticker].fillna(0) * allocation / 100
            else:
                print(f"Warning: Allocation for {ticker} is None.")

    blended_portfolio_df = pd.DataFrame(blended_portfolio, columns=['blended_portfolio'])
    blended_portfolio_df['norm_close'] = blended_portfolio_df['blended_portfolio']
    
    print(f"Bad tickers: {bad_tickers}")
    return blended_portfolio_df, events

def parse_portfolio_data(text):
    # remove comment from the input string
    text = re.sub(r"//.*", "", text)

    # find the text inside the brackets
    data_text = re.search(r'\[(.*)\]', text, re.DOTALL).group(1)

    # turn it back into a python list of dictionaries
    data = ast.literal_eval('[' + data_text + ']')
    
    return data

@app.route('/api/scenarios', methods=['POST'])
def api_scenarios():
    data = request.json
    query = data.get('query')
    username = data.get('username', 'default_user')
    print("Request data:", data)
    portfolioName = data.get('portfolioName', 'default_portfolio')
    
    if portfolioName != 'default_portfolio':
        loaded_portfolio = load_user_portfolio_from_dynamodb(username, portfolioName)
        loaded_portfolio = {investment_key: {'allocation': investment_data['allocation']}
                       for investment in loaded_portfolio['Investments']
                       for investment_key, investment_data in investment.items()}

        print('loaded db portfolio', loaded_portfolio, 'default', default_portfolio)    
    else:
        loaded_portfolio = default_portfolio
        print('loaded default portfolio', loaded_portfolio)

    # received prompt data
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    gpt_prompt = f"""given this query: {query}"""

    print("scenarios:: gpt_prompt:", gpt_prompt)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "please list the major market impacting events given the users' query along with a detailed date including year-month-day. response must be in a dictionary format like so: [{'event': 'event 1', 'date': '2020-02-12', event_sent: 0.88'}, {'event': 'event 2', 'date': '2021-12-31', event_sent: 0.28}]",
            },
            {
                "role": "user", 
                "content": gpt_prompt
            },
        ],
    )

    gpt_response = response.choices[0].message['content'].strip()
    print("scenarios:: trianglai_response:", gpt_response)
    # Use 'gpt_response' instead of 'text' as input for fetch_data function

    data, events = fetch_data(loaded_portfolio, text=gpt_response)
    print("scenarios:: fetched_data:", data)

    # summarize
    gpt_prompt_story = f"""summarize this simply and concisely: {query}"""
    
    response_story = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "you help users understand their data better, by providing incredible summaries.",
            },
            {
                "role": "user", 
                "content": gpt_prompt_story
            },
        ],
    )

    gpt_response_story = response_story.choices[0].message['content'].strip()
    print("scenarios:: trianglai_response_story:", gpt_response_story)

    # Save chat history to DynamoDB
    chat = {
        "query": query,
        "trianglai_response": gpt_response_story,
        "trianglai_response_insights": gpt_response,
    }
    save_chat_history(username, timestamp, chat)

    # Convert pandas dataframes to json before returning
    data_json = {ticker: df.to_json() for ticker, df in data.items()}
    return jsonify({'trianglai_response': gpt_response_story, 'username': username, 'data': data_json, 'events': events})

### SCENARIOS - END ###

### FORESIGHT - START ###

### FORESIGHT - END ###

### PORTFOLIO STATE MGMT ###
def get_portfolio_names(user_id):

    response = table_portfolio.scan(
        FilterExpression=Attr('UserId').eq(user_id)
    )
    
    portfolio_names = [item['PortfolioName'] for item in response['Items']]
    
    return portfolio_names

@app.route('/api/portfolios/<user_id>')
def portfolios(user_id):
    portfolio_names = get_portfolio_names(user_id)
    return jsonify(portfolio_names)

@app.route('/api/getPortfolio', methods=['GET'])
def api_get_portfolio():
    # Get the userId and portfolioName from the request arguments
    username = request.args.get('userId')
    portfolio_name = request.args.get('portfolioName')

    if username is None or portfolio_name is None:
        return jsonify({"error": "userId and portfolioName are required"}), 400

    try:
        # Fetch the portfolio from DynamoDB
        response = table_portfolio.get_item(
            Key={
                'UserId': username,
                'PortfolioName': portfolio_name,
            }
        )

        # If the portfolio exists, return it
        if 'Item' in response:
            # Create a dictionary for the DynamoDB load
            dynamodb_load = {
                "UserId": username,
                "PortfolioName": portfolio_name,
                "Investments": response['Item']['Investments']
            }
            
            print("load portfolio::: ", dynamodb_load)
            return jsonify(dynamodb_load), 200
        else:
            return jsonify({"error": "Portfolio not found"}), 404

    except Exception as e:
        # Log the error message and the stack trace
        app.logger.error('Server Error: %s', str(e))
        traceback.print_exc()
        # If there is an error, return a error message and a 500 status
        return jsonify({"error": str(e)}), 500
### PORTFOLIO STATE MGMT - END ###

### PORTFOLIO UPDATE FETCH ###
@app.route('/api/portfolioUpdate', methods=['POST'])
def portfolio_update():
    data = request.get_json()  # Get data sent in the POST request

    # Get individual data
    username = data.get('username')
    uid = data.get('uid')
    portfolioName = data.get('portfolioName', 'default_portfolio')
    eventData = data.get('eventData')

    # Print the received data
    print(f'Username: {username}')
    print(f'UID: {uid}')
    print(f'Portfolio Name: {portfolioName}')
    print(f'Event Data: {eventData}')

    if portfolioName != 'default_portfolio':
        loaded_portfolio = load_user_portfolio_from_dynamodb(uid, portfolioName)
        print('loaded db portfolio', loaded_portfolio)    
    else:
        loaded_portfolio = default_portfolio
        print('loaded default portfolio')

    response = {
        'status': 'success',
        'message': 'Portfolio updated successfully'
    }

    return json.dumps(response), 200
### PORTFOLIO UPDATE FETCH - END ###

@app.route('/log-all', methods=['POST'])
def log_all_requests():
    print(request.data)
    return jsonify({"message": "Request logged."}), 200

### STRIPE SESSION ###
# This is your Stripe CLI webhook secret for testing your endpoint locally.
endpoint_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
YOUR_DOMAIN = "http://localhost:3000" #"https://cuanto.io"  # Replace with your website domain

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    plan_type = request.json.get('plan')
    
    if plan_type not in ['annual', 'monthly']:
        return jsonify({"error": "Invalid plan type"}), 400

    product_price = {
        'annual': 'price_1NeKGSBhBxXSh10sptHpz205',
        'monthly': 'price_1NgMnFBhBxXSh10sQbjxbAb6'
    }

    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{
            'price': product_price[plan_type],
            'quantity': 1,
        }],
        mode='subscription',
        success_url=YOUR_DOMAIN + "/scenarios?session_id={CHECKOUT_SESSION_ID}",
        cancel_url=YOUR_DOMAIN + "/cancel",
    )

    print(f"Created stripe checkout session for plan: {plan_type}. Session ID: {session.id}")

    # Process payment
    data = {'session_id': session.id}
    process_payment(data)

    return jsonify({"session_url": session.url})

# @app.route('/payment-endpoint', methods=['POST'])
# def handle_payment_confirmation():
#     session_id = request.json.get('sessionId')
#     return jsonify({"success": True, "message": "Payment confirmed!"})

@app.route('/payment-endpoint', methods=['POST'])
def handle_payment_confirmation():
    session_id = request.json['sessionData'].get('id')
    userId = request.json.get('userId')
    userEmail = request.json.get('userEmail')

    if not session_id:
        return jsonify({"success": False, "message": "No sessionId provided!"})

    try:
        # Fetch the checkout session
        session = stripe.checkout.Session.retrieve(session_id)
        print("ISSUE 9:: session::", session)
        print("ISSUE 9:: good 1")
        # Insert into DynamoDB
        response = table_payments.put_item(
            Item={
                'userId': userId,
                'userEmail': userEmail,
                'paymentId': session.invoice,
                'timestamp': str(session.created),
                'amountTotal': session.amount_total,
                'currency': session.currency,
                'customerId': session.customer,
                'email': session.customer_details['email'],
                'event_status': session.status,  # or session.status based on what's appropriate
                'invoiceId': session.invoice,
                'paymentStatus': session.payment_status,
                'session_id': session.id,
                'session_type': session.object,
                'subscriptionId': session.subscription
            }
        )

        print("ISSUE 9:: good 2")
        # You can check the response, if needed, for additional verification
        
        return jsonify({
            "success": True,
            "message": "Payment confirmed and data stored!"
        })

    except stripe.error.StripeError as e:
        return jsonify({"success": False, "message": f"Stripe error: {str(e)}"})
    except Exception as e:  # Catching DynamoDB related errors
        return jsonify({"success": False, "message": f"Database error: {str(e)}"})

def process_payment(data):
    session_id = data.get("session_id")
    paymentId = session_id
    if not session_id:
        message = "Session ID is missing."
        print(message)
        return

    current_time = datetime.utcnow().isoformat()
    response = table_payments.put_item(
        Item={
            'session_type': 'process_payment',
            'session_id': session_id,
            'paymentId': paymentId,
            'timestamp': current_time,            
        }
    )
    print("process_payment response:", response)
    return response
### STRIPE SESSION - END ###

@app.route('/check-isPremiumUser', methods=['POST'])
def check_is_premium_user():
    user_id = request.json.get('userId')
    user_email = request.json.get('userEmail')

    # Fetch payment info from DynamoDB
    response = table_payments.scan(
        FilterExpression=(boto3.dynamodb.conditions.Attr('userId').eq(user_id) & 
                        boto3.dynamodb.conditions.Attr('userEmail').eq(user_email) &
                        boto3.dynamodb.conditions.Attr('paymentStatus').eq('paid') &
                        boto3.dynamodb.conditions.Attr('event_status').eq('complete'))
    )

    userItems = response.get('Items')

    # Loop through the user items and check if any item matches the criteria
    for item in userItems:
        if item['paymentStatus'] == 'paid' and item['event_status'] == 'complete':
            return jsonify({"isPremiumUser": True})

    return jsonify({"isPremiumUser": False})

if __name__ == "__main__":
    app.run(port=5000)
    # app.run(host="0.0.0.0", port=8080)