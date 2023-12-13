# support api
from datetime import datetime, timedelta, date
from boto3.dynamodb.conditions import Key, Attr
from joblib import Parallel, delayed
from dotenv import load_dotenv
from scipy.stats import norm
# from prophet import Prophet
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

from io import StringIO

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
S3_BUCKET = 'tri-cfo-uploads'
CHAT_HISTORY_PREFIX = 'chat_history/'

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
table = dynamodb.Table(os.getenv("DYNAMODB_TABLE_INPUT_PROMPTS"))
table_portfolio = dynamodb.Table(os.getenv("DYNAMODB_TABLE_PORTFOLIO"))
table_user_profiles = dynamodb.Table(os.getenv("DYNAMODB_TABLE_USER_PROFILE"))
table_chat_history = dynamodb.Table(os.getenv("DYNAMODB_TABLE_CHAT_HISTORY"))
table_payments = dynamodb.Table(os.getenv("DYNAMODB_TABLE_PAYMENTS")) 

table_TriDB = dynamodb.Table(os.getenv("DYNAMODB_TABLE_TRIDB")) 
table_metrics = dynamodb.Table(os.getenv("DYNAMODB_TABLE_METRICS")) 

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
    return 'eb-live alpha tri v3.06'

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

@app.route('/api/add-manual-portfolio', methods=['POST'])
def add_manual_portfolio():
    data = request.json
    userId = data.get('userId', 'noname')
    portfolio_name = data.get('username', 'Manual Portfolio')
    portfolio = data.get('portfolio')

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    dynamodb_entry = {
        "UserId": userId,
        "PortfolioName": f"{portfolio_name} - {timestamp}",
        "Portfolio": portfolio,
        "Feature": "manual add",
    }

    #save portfolio
    print("new feat::: dynamodb_entry:", dynamodb_entry)
    save_user_portfolio_to_dynamodb(dynamodb_entry)

    return jsonify(dynamodb_entry), 200    

def save_user_portfolio_to_dynamodb(dynamodb_entry):
    """
    Add a user's manual portfolio to a DynamoDB table.
    """

    print("dynamodb_entry::", dynamodb_entry)
    table_portfolio.put_item(
        Item={
            'UserId': dynamodb_entry["UserId"],
            'Portfolio': json.dumps(dynamodb_entry["Portfolio"]),
            'PortfolioName': dynamodb_entry["PortfolioName"],
            'Feature': dynamodb_entry["Feature"],
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

    n_sims = 10000
    risk_free_rate = 0.045
    num_portfolios = n_sims

    # results, weights_record = simulate_portfolios_parallel(num_portfolios, mean_returns, cov_matrix, risk_free_rate)
    results, weights_record, risk_scores, return_scores = simulate_portfolios_parallel(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_index = results[2, :].argmax()
    min_vol_index = results[1, :].argmin()

    max_sharpe_allocation, min_vol_allocation = get_key_allocations(results, weights_record, stock_data)
    
    portfolio_allocation = max_sharpe_allocation
    print(f"print portfolio_allocation: {portfolio_allocation.to_dict()}")
    blended_portfolio, events = fetch_data(portfolio_allocation.to_dict(), '')
    data_json = {ticker: df.to_json() for ticker, df in blended_portfolio.items()}

    # save portfolio to chat history v1.06
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    portfolio_summary = format_data(portfolio_allocation)

    chat = {
        "query": prompt_tickers,
        "trianglai_response": portfolio_summary
    }
    save_chat_history(username, timestamp, chat)
    
    dynamodb_entry = {
        "UserId": username,
        "PortfolioName": f"Gen AI Portfolio - {timestamp}",
        "Portfolio": portfolio_allocation.to_dict(),
        "Feature": "portfolios",
    }

    #save portfolio
    print("new feat::: dynamodb_entry:", dynamodb_entry)
    save_user_portfolio_to_dynamodb(dynamodb_entry)

    # print('risk_score', risk_score,'return_score', return_score)

    return jsonify({
        'data': data_json, 
    })

@app.route('/api/combined_summary', methods=['POST'])
def api_combined_summary():
    data = request.json
    query = data.get('query')
    num_results = data.get('num_results', 10)
    username = data.get('username', 'tsm')

    portfolioName = data.get('portfolioName', 'default_portfolio')
    
    if portfolioName != 'default_portfolio':
        loaded_portfolio = load_user_portfolio_from_dynamodb(username, portfolioName)

        # Ensure 'Portfolio' is a dictionary and that it contains items
        if 'Portfolio' in loaded_portfolio and isinstance(loaded_portfolio['Portfolio'], dict):
            processed_portfolio = {}
            for investment_key, investment_data in loaded_portfolio['Portfolio'].items():
                if isinstance(investment_data, dict) and 'allocation' in investment_data:
                    processed_portfolio[investment_key] = {'allocation': investment_data['allocation']}
            loaded_portfolio = processed_portfolio
        else:
            loaded_portfolio = {}

        print('loaded db portfolio', loaded_portfolio, 'default', default_portfolio)    
    else:
        loaded_portfolio = default_portfolio
        print('loaded default portfolio', loaded_portfolio)

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
                "content": "You're an AI that chats in very short responses. You must answer as best as you can. Your objective is gain more trust. As a portfolio management co-pilot you're responses assist with investment strategy, ticker recommendations, portfolio optimization and other investment queries. ",
            },
            {
                "role": "user", 
                "content": gpt_prompt + " my portfolio: " + str(loaded_portfolio)
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

PROFILE_PREFIX = 'user_profile/'

def save_user_portfolio_to_dynamodb(dynamodb_entry):
    """
    Save a user's portfolio and temporary portfolio to a DynamoDB table.
    """

    print("dynamodb_entry::", dynamodb_entry)

    table_portfolio.put_item(
        Item={
            'UserId': dynamodb_entry["UserId"],
            'Portfolio': json.dumps(dynamodb_entry["Portfolio"]),
            'PortfolioName': dynamodb_entry["PortfolioName"],
            'Feature':  dynamodb_entry["Feature"],        
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

        Portfolio = json.loads(item.get('Portfolio', '{}'))

        return {
            'UserId': UserId,
            'Portfolio': Portfolio,
            'PortfolioName': PortfolioName,
        }

### SCENARIOS - START ###
import yfinance as yf
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
import concurrent.futures

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

        # Ensure 'Portfolio' is a dictionary and that it contains items
        if 'Portfolio' in loaded_portfolio and isinstance(loaded_portfolio['Portfolio'], dict):
            processed_portfolio = {}
            for investment_key, investment_data in loaded_portfolio['Portfolio'].items():
                if isinstance(investment_data, dict) and 'allocation' in investment_data:
                    processed_portfolio[investment_key] = {'allocation': investment_data['allocation']}
            loaded_portfolio = processed_portfolio
        else:
            loaded_portfolio = {}

        print('loaded db portfolio', loaded_portfolio, 'default', default_portfolio)    
    else:
        loaded_portfolio = default_portfolio
        print('loaded default portfolio', loaded_portfolio)

    # received prompt data
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    gpt_prompt = f"""given this query: {query}"""

    print("scenarios:: gpt_prompt:", gpt_prompt)

    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "please list the major market impacting events given the users' query along with a detailed date including year-month-day. response must be in a dictionary format like so: [{'event': 'event 1', 'date': '2020-02-12', event_sent: 0.88'}, {'event': 'event 2', 'date': '2021-12-31', event_sent: 0.28}]",
            },
            {
                "role": "user", 
                "content": gpt_prompt + " for this portfolio: " + str(loaded_portfolio)
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
    return jsonify({
        'trianglai_response': gpt_response_story, 
        'username': username, 
        'data': data_json, 
        'events': events
        })

### SCENARIOS - END ###

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
                "Portfolio": response['Item']['Portfolio']
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
YOUR_DOMAIN = "https://triangl.ai" #"https://cuanto.io"  # "http://localhost:3000" #Replace with your website domain

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    plan_type = request.json.get('plan')
    
    if plan_type not in ['annual', 'monthly']:
        return jsonify({"error": "Invalid plan type"}), 400

    product_price = {
        'annual': 'price_1Nh0txBhBxXSh10sUTPRHczd',
        'monthly': 'price_1Nh0txBhBxXSh10sHsx4SGLo'
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

from decimal import Decimal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

### Tri ###
class Collection:
    def __init__(self, user_id):
        self.user_id = user_id        
        self.table = table_TriDB  # DynamoDB table name
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.embeddings = np.array([])  # Initialize as a NumPy array for consistency

    def fit_vectorizer_with_dynamodb(self):
        # Fetch documents from DynamoDB
        response = self.table.scan()
        documents = [item['document'] for item in response['Items'] if item['UserId'] == self.user_id]

        if not documents:
            raise ValueError("No documents to fit the vectorizer.")
        self.vectorizer.fit(documents)

    def add_to_dynamodb(self, documents=[], metadatas=[], timestamps=[]):
        if not documents:
            return  # Nothing to add

        for doc, metadata, timestamp in zip(documents, metadatas, timestamps):
            try:
                # Assuming metadata is already in a format that DynamoDB can accept
                self.table.put_item(
                    Item={
                        'UserId': self.user_id,  # Make sure this is a string
                        'document': doc,  # Ensure this is a string
                        'metadata': metadata,  # Directly passed without json.dumps
                        'timestamp': timestamp  # Ensure this is in the correct format
                    }
                )
                # Consider adding logging here for successful operations
            except Exception as e:
                # Log the error
                print(f"Error adding document: {e}")
        
        # Update the vectorizer
        self.fit_vectorizer_with_dynamodb()

    def query_from_dynamodb(self, query_texts, n_results=1):
        if not query_texts:
            return []

        # Fetch and transform documents from DynamoDB
        response = self.table.scan(
            FilterExpression=Key('UserId').eq(self.user_id)
        )
        documents = [item['document'] for item in response['Items']]

        if not documents:
            return []

        # Transform documents using the already fitted vectorizer
        self.fit_vectorizer_with_dynamodb()  # Ensure vectorizer is up-to-date
        document_embeddings = self.vectorizer.transform(documents).toarray()
        query_embedding = self.vectorizer.transform(query_texts)

        # Compute similarities with each document
        cosine_similarities = linear_kernel(query_embedding, document_embeddings).flatten()

        # Get indices of top n_results similar documents
        top_indices = cosine_similarities.argsort()[:-(n_results+1):-1]
        top_indices = top_indices[:min(len(documents), n_results)]

        # Return top documents using the top_indices
        return [documents[i] for i in top_indices]

### -tri- ###
def context_classifier(content):

    tags = ['chat', 'person', 'organization', 'action', 'object', 'etc']
    
    try:
        context_classifier = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"turn inputs to keys and values. available tags: {tags} format must be: [['content','value'],['action','value']]."
                    # "content": "turn inputs to keys and values. respond with dict only. format must be: [{'content': content, 'tags': chat, action, timeline, object etc.}]"
                },
                {
                    "role": "user", 
                    "content": content
                },
            ],
        )

        context_layers = context_classifier.choices[0].message['content'].strip()
        print(context_layers)

        # Use regex to find the list-like pattern in the string
        match = re.search(r"\[\[.*?\]\]", context_layers)
    except: 
        match = None
    
    if match:
        list_string = match.group(0)
        try:
            # Convert the extracted string to an actual Python list
            actual_list = ast.literal_eval(list_string)
            return actual_list
        except (ValueError, SyntaxError):
            # Handle the case where the string is not a valid Python literal
            return "Invalid list format"
    else:
        return "No list found in string"

def pdGPT(prompt):
    """
    This function takes a prompt and uses GPT-3.5 to generate Pandas code.
    """
    try:
        print('inside pdGPT')

        response = openai.ChatCompletion.create(
            model= "ft:gpt-3.5-turbo-1106:triangleai::8VE9dhcM", #"gpt-3.5-turbo",
            # model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "respond with full python code from import to final string response. only code.",
                },
                {
                    "role": "user", 
                    "content": prompt
                },
            ],
        )

        gpt_response = response.choices[0].message['content'].strip()
        print('gpt python:', gpt_response, 'end')
        return gpt_response

    except Exception as e:
        return f"An error occurred: {e}"

def get_files_from_s3(folder):
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder)
    files = [item['Key'] for item in response.get('Contents', [])]
    return files

@app.route('/lists', methods=['GET'])
def get_list():
    button_name = request.args.get('button', '')
    if button_name:
        # folder_name = f'{button_name}/' 
        folder_name = ''
        files = get_files_from_s3(folder_name)
        return jsonify(files)
    else:
        return jsonify([]), 404

@app.route('/tri', methods=['POST'])
def triChat():
    print('tri - live')
    # Get text fields from form data
    userId = request.form.get('username', 'noname')
    gpt_prompt = request.form.get('prompt')
    feature = request.form.get('feature')
    selectedFile = request.form.get('activeFile')
    print(feature, selectedFile)

    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Access uploaded files
    uploaded_files = request.files.getlist('files')
    filename_list = []
    for file in uploaded_files:
        print('file: ', file.filename)     
        filename_list.append(file.filename)
        if file.filename.endswith('.xlsx'):
            # Handle Excel files
            df = pd.read_excel(file, engine='openpyxl')
        elif file.filename.endswith('.csv'):
            # Handle CSV files
            df = pd.read_csv(file)
        else:
            # Skip unsupported file types
            print(f"Unsupported file type: {file.filename}")
            continue

        print(timestamp, 'file read', file.filename, df.head())
    
        # Reset the file's read pointer to the start
        file.seek(0)
        
        # Now upload to S3
        upload_file_to_s3(file, S3_BUCKET)

    if feature == 'Forecast':
        gpt_response = 'Forecasts are coming soon!'
        return jsonify(gpt_response)

    if feature == 'Analyze': 
        print('here')
        object_key = selectedFile

        you_are = "Hi, I'm Tri. I help you make more decicive action based on all context given"

        def read_csv_from_s3(bucket_name, object_key):
            s3 = boto3.client('s3')
            csv_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
            body = csv_obj['Body']
            csv_string = StringIO(body.read().decode('utf-8'))
            df = pd.read_csv(csv_string)
            return df

        # Use the function to read CSV and print the first few lines
        df = read_csv_from_s3(S3_BUCKET, object_key)        

        pandas_task = f"""# given df is loaded. df.head() = {df.head()} write python to execute based on this request: {gpt_prompt}"""
        python_code = pdGPT(pandas_task)

        print("python_code:", python_code, "end")

        # Make sure df is accessible in the local scope of exec
        local_vars = {'df': df}
        exec(python_code, globals(), local_vars)

        # Retrieve the result from local_vars
        python_result = local_vars['result']
        try:
            python_result_comment = local_vars['result_comment']
            print("python_result:", str(python_result) + str(python_result_comment), "end")
        except:
            python_result_comment = ''
            print("python_result:", str(python_result) + str(python_result_comment), "end")

        # response
        response = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-1106:triangleai::8U4LPTy8", #"gpt-3.5-turbo", #"gpt-4-1106-preview"
            messages=[
                {
                    "role": "system",
                    "content": f"I am: {you_are}."
                },
                {
                    "role": "user", 
                    "content": f"{gpt_prompt} \ndata analysis: {str(python_result) + str(python_result_comment)}"
                },
            ],
        )

        gpt_response = response.choices[0].message['content'].strip()
        print(gpt_response)

        return jsonify(gpt_response + '\n\nResults:\n' + str(python_result) + str(python_result_comment))
        
    print(userId)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure the collection is retrieved or created
    collection = Collection(userId)

    filenames = ', '.join(filename_list)

    if uploaded_files:
        gpt_prompt = gpt_prompt + 'uploaded files:'+ filenames
        context_layers = context_classifier(gpt_prompt + 'uploaded files:'+ filenames)
    else:
        # context classifier
        context_layers = context_classifier(gpt_prompt)

    try:
        # add context key and values
        for i in range(len(context_layers)):

            context_key = context_layers[i][0]
            context_value = context_layers[i][1]

            collection.add_to_dynamodb(
                documents=[context_value],
                metadatas=[context_key],
                timestamps=[timestamp],
            )        

            # multi-context search 
            search_results = []

            for i in range(len(context_layers)):

                context_key = context_layers[i][0]

                results = collection.query_from_dynamodb([gpt_prompt + context_key], n_results=3)

            search_results.append(results)
            print('search_results',search_results,'\n')
        
    except:
        unique_strings = ['']

    # full retrieved context
    # Flatten the list of lists and remove duplicates using a set
    unique_strings = set(item for sublist in search_results for item in sublist)

    # Join the unique strings
    joined_string = ' '.join(unique_strings)
    
    # full context
    if userId == 'Alpha':
        userId = "Steve"
    full_context = f"{timestamp}. past chats wit me: {joined_string}. you: {gpt_prompt}."

    collection.add_to_dynamodb(
        documents=[full_context],
        metadatas=[unique_strings],
        timestamps=[timestamp],
    )        

    you_are = "Hi, I'm Tri. I help you make more decicive action based on all context given"

    # response
    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-1106:triangleai::8U4LPTy8", #"gpt-3.5-turbo", #"gpt-4-1106-preview"
        messages=[
            {
                "role": "system",
                "content": f"I am: {you_are}."
            },
            {
                "role": "user", 
                "content": full_context
            },
        ],
    )

    gpt_response = response.choices[0].message['content'].strip()
    print(gpt_response)

    # context classifier
    context_layers_response = context_classifier(gpt_response)

    try:
        # add context key and values
        for i in range(len(context_layers_response)):

            context_key = context_layers_response[i][0]
            context_value = context_layers_response[i][1]

            collection.add_to_dynamodb(
                documents=[context_value],
                metadatas=[context_layers_response],
                timestamps=[timestamp],
            ) 
    except:
        print('context layers issue')

    return jsonify(gpt_response)
### -tri- ###

### -upload- ###
def upload_file_to_s3(file, bucket_name):
    try:
        filename = secure_filename(file.filename)
        s3.upload_fileobj(
            file,
            bucket_name,
            filename,
            ExtraArgs={
                "ContentType": file.content_type  # Make sure no ACL-related arguments are here
            }
        )
        # Presigned URL generation remains the same
        presigned_url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': filename}, ExpiresIn=3600)
        return presigned_url
    except FileNotFoundError:
        return jsonify({"error": "The file was not found"}), 404
    except NoCredentialsError:
        return jsonify({"error": "Credentials not available"}), 403
    except Exception as e:
        # Catch any other exception and return a meaningful error message
        return jsonify({"error": str(e)}), 500        

### -cta- ###
from werkzeug.utils import secure_filename

@app.route('/metrics', methods=['POST'])
def collect_metrics():
    data = request.json
    interactions = data.get('interactions', [])
    
    print("Received data:", data)

    for interaction in interactions:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        userId = interaction.get('username', 'noname')
        metric_name = interaction.get('metric_name', 'nometric')

        try:
            # Insert the metric data into the DynamoDB table
            table_metrics.put_item(
                Item={
                    'metric_name': metric_name,
                    'timestamp': timestamp,
                    'userId': userId
                }
            )

        except ClientError as e:
            print(e.response['Error']['Message'])
            return jsonify({'message': 'Error recording metric'}), 500

        except Exception as e:
            print(str(e))
            return jsonify({'message': 'Internal server error'}), 500

    return jsonify({'message': 'Metrics recorded successfully'}), 201

### -cta- ###

if __name__ == "__main__":
    # app.run(port=5000)
    app.run(host="0.0.0.0", port=8080)