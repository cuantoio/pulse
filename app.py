# support api
from datetime import datetime, timedelta, date
from boto3.dynamodb.conditions import Key
from joblib import Parallel, delayed
from dotenv import load_dotenv
from scipy.stats import norm
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
CORS(app)

load_dotenv()

openai.organization = "org-bHNvimGOVpNGPOLseRrHQTB4"
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    return 'eb-live v1.07'

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
    user_portfolio = load_user_portfolio_from_dynamodb(username)

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

PROFILE_PREFIX = 'user_profile/'

def save_user_portfolio_to_dynamodb(portfolio, username):
    """
    Save a user's portfolio and temporary portfolio to a DynamoDB table.
    """

    print("portfolio + username::", portfolio, username)

    table_portfolio.put_item(
        Item={
            'username': username,
            'portfolio': json.dumps(portfolio),
        }
    )

def load_user_portfolio_from_dynamodb(username):
    """
    Load a user's portfolio from DynamoDB. 
    If the user has a saved portfolio, load it as the temporary portfolio.
    """
    try:
        response = table_portfolio.get_item(Key={'username': username})
        print("load user")
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        item = response.get('Item')
        if item is None:
            return None

        portfolio = json.loads(item.get('portfolio', '{}'))
        temp_portfolio = json.loads(item.get('temp_portfolio', '{}'))

        return {
            'username': username,
            'portfolio': portfolio,
        }

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
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[1])
    sdp_min, rp_min = results[1, min_vol_idx], results[0, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights_record[min_vol_idx], index=stock_data.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
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

@app.route('/api/efficient_frontier', methods=['POST'])
def efficient_frontier():
    data = request.json
    username = data.get('username', 'tsm')
    prompt = data.get('query', '').lower()

    print(f"Data received: {data}")

    prompt_tickers = data.get('tickers', '')

    user_profile = load_user_portfolio_from_dynamodb(username)
    print("user_profile::",user_profile)

    # Parse tickers from the prompt
    # prompt_tickers = prompt.split('$')[1].split()    
    # prompt_tickers = [ticker.upper() for ticker in prompt_tickers]

    print(f"Prompt tickers after parsing and conversion: {prompt_tickers}")

    # Create an empty dictionary for the loaded user portfolio
    loaded_user_portfolio = {}

    for ticker in prompt_tickers:
        if user_profile is None or ticker not in user_profile:
            loaded_user_portfolio[ticker] = {"allocation": 0}  # default allocation

    print(f"Loaded user portfolio: {loaded_user_portfolio}")

    tickers = list(loaded_user_portfolio.keys())
    print(f"Final list of tickers: {tickers}")

    allocations = [v["allocation"] for v in loaded_user_portfolio.values()]

    stock_data_key = tuple(sorted(tickers))

    # Use today's date as the end date and one year ago as the start date
    today = date.today()
    start_date = today - timedelta(days=365)
    end_date = today

    if stock_data_key not in stock_data_cache:
        try:
            stock_data_cache[stock_data_key] = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        except:
            print(stock_data_key, "failed")

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

    # save portfolio to chat history v1.06
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    portfolio_summary = format_data(portfolio_allocation)

    chat = {
        "query": prompt_tickers,
        "response": portfolio_summary
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

    user_portfolio = load_user_portfolio_from_dynamodb(username)

    # Parse tickers from the prompt
    # query_tickers = []
    # if '$' in query:
    #     query_tickers = query.split('$')[1].split()
    #     query_tickers = [ticker.upper() for ticker in query_tickers]

    if query:
        if 'price' in query or 'prices' in query:
            print('price request')

        # Load last chat history
        last_chat = load_chat_history(username)
        if last_chat:
            # If a previous chat history exists, add it to the prompt
            previous_query = last_chat['query']
            previous_response = last_chat['response']
            gpt_prompt = f"Last query: {previous_query}\nLast response: {previous_response}\n\nCurrent query: {query}\n\nPlease provide a summary for the following query in a conversational tone: {query}. If the query involves analyzing the optimized portfolio, the current portfolio is: {user_portfolio}"
        else:
            gpt_prompt = f"Please respond in short form, maintain conversational tone: {query}. If the query involves analyzing the optimized portfolio, the current portfolio is: {user_portfolio}" 

        response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model="gpt-4",
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
            "response": gpt_response
        }
        save_chat_history(username, timestamp, chat)

        return jsonify({'combined_summary': gpt_response, 'username': username})

    else:  # If the user query is empty, ask for more details.
        combined_summary = "Could you give us more specifics on what you're intending to accomplish?" 
        return jsonify({'combined_summary': combined_summary, 'username': username})

### SCENARIOS - START ###
import yfinance as yf
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
import concurrent.futures

default_portfolio = {
    # 'EEM': {'allocation': 4.09}, 
    # 'EFA': {'allocation': 25.94}, 
    # 'IEF': {'allocation': 0.85}, 
    # 'IJH': {'allocation': 23.83}, 
    # 'IJR': {'allocation': 26.2}, 
    # 'LQD': {'allocation': 0.29}, 
    'SPY': {'allocation': 3.82}, 
    # 'VNQ': {'allocation': 14.99}
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

def download_ticker_data(ticker, start_date, end_date):
    ticker_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')
    ticker_data['norm_close'] = ticker_data['Close'].div(ticker_data['Close'].iloc[0]).subtract(1).mul(100)
    return ticker, ticker_data

def fetch_data(portfolio, text):
    events = extract_events_dates(text)
    event_dates = sorted([parse_date(date) for date in events.keys()])
    start_date = event_dates[0] - relativedelta(months=6)
    end_date = event_dates[-1] + relativedelta(months=6)

    tickers = list(portfolio.keys())

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_ticker = {executor.submit(download_ticker_data, ticker, start_date, end_date): ticker for ticker in tickers}
        data = {}
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                data[ticker] = future.result()[1]
            except Exception as exc:
                print('%r generated an exception: %s' % (ticker, exc))

    return data, events

@app.route('/api/scenarios', methods=['POST'])
def api_scenarios():
    data = request.json
    query = data.get('query')
    username = data.get('username', 'tsm')

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
                "content": "please list the major market impacting events given the users' query along with a detailed date including year-month-day. response must be in a dictionary format like so: [{'event': 'event 1', 'date': '2020-02-12'}, {'event': 'event 2', 'date': '2021-12-31'}]",
            },
            {
                "role": "user", 
                "content": gpt_prompt
            },
        ],
    )

    gpt_response = response.choices[0].message['content'].strip()
    print("scenarios:: scenarios_response:", gpt_response)
    # Use 'gpt_response' instead of 'text' as input for fetch_data function
    data, events = fetch_data(default_portfolio, text=gpt_response)

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
    print("scenarios:: scenarios_response_story:", gpt_response_story)

    # Save chat history to DynamoDB
    chat = {
        "query": query,
        "scenarios_response_story": gpt_response_story,
        "gpt_response": gpt_response,
    }
    save_chat_history(username, timestamp, chat)

    # Convert pandas dataframes to json before returning
    data_json = {ticker: df.to_json() for ticker, df in data.items()}
    return jsonify({'scenarios_response': gpt_response_story, 'username': username, 'data': data_json, 'events': events})

### SCENARIOS - END ###

if __name__ == "__main__":
    # app.run(port=5000)
    app.run(host="0.0.0.0", port=8080)
