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
    return 'eb-live alpha tri v3.13'

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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import linear_kernel

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

### START - Forecast Feature ###
def analyze_dataframe(df):
    results = {
        "date_time_columns": [],
        "numerical_columns": []
    }

    for column in df.columns:
        # Attempt to convert column to datetime if it's not already
        if df[column].dtype == 'object':
            try:
                df[column] = pd.to_datetime(df[column])
            except ValueError:
                pass  # If conversion fails, continue as normal

        # Check for date, datetime, or time columns
        if pd.api.types.is_datetime64_any_dtype(df[column]) or pd.api.types.is_timedelta64_dtype(df[column]):
            sorted_col = df[column].sort_values()
            first_value = sorted_col.iloc[0]
            last_value = sorted_col.iloc[-1]
            results["date_time_columns"].append({
                "column_name": column,
                "type": df[column].dtype.name,
                "first_value": first_value,
                "last_value": last_value
            })

        # Check for numerical (int or float) columns
        elif pd.api.types.is_numeric_dtype(df[column]):
            results["numerical_columns"].append(column)

    return results

def predGPT(prompt):
    gpt = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-1106:triangleai::8U4LPTy8",
        messages=[
            {
                "role": "system",
                "content": f"write a short summary of what data we have for analysis and prediction. recommend prediction ml model. respond natural language."
            },
            {
                "role": "user", 
                "content": prompt
            },
        ],
    )

    response = gpt.choices[0].message['content'].strip()

    return response

### END - Forecast Feature ###

def pdGPT(prompt):
    """
    This function takes a prompt and uses GPT-3.5 to generate Pandas code.
    """
    try:
        print('inside pdGPT')

        response = openai.ChatCompletion.create(
            model= "ft:gpt-3.5-turbo-1106:triangleai::8XFC4wE8", #"ft:gpt-3.5-turbo-1106:triangleai::8X4SYg4T", #"ft:gpt-3.5-turbo-1106:triangleai::8VQe5IQ9", #"ft:gpt-3.5-turbo-1106:triangleai::8VE9dhcM", #"gpt-3.5-turbo",
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
    S3_BUCKET = 'tri-cfo-uploads'

    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder)
    files = [item['Key'] for item in response.get('Contents', [])]
    return files

def read_csv_from_s3(bucket_name, folder_name, file_name):
    # Construct the full object key with folder and file name
    object_key = f"{folder_name}/{file_name}"

    # Get the object from S3
    csv_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
    
    # Read the object's body into a DataFrame
    body = csv_obj['Body']
    csv_string = StringIO(body.read().decode('utf-8'))
    df = pd.read_csv(csv_string)
    
    return df

def read_json_from_s3(bucket_name, folder_name, file_name, prompt):
    
    object_key = f"{folder_name}/{file_name}"
    response = s3.get_object(Bucket=bucket_name, Key=object_key)

    # Read the content of the file and parse it as JSON
    ent_data = json.loads(response['Body'].read())

    combined_texts = [f"timestamp: {data['timestamp']} input: {data['input']} output: {data['output']}" for data in ent_data]

    # Vectorizing text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    # Using KNN to find similar movies
    knn = NearestNeighbors(n_neighbors=3, metric='cosine')
    knn.fit(tfidf_matrix)

    query_tfidf = vectorizer.transform([prompt])

    # Find similar movies
    distances, indices = knn.kneighbors(query_tfidf, n_neighbors=3)
    
    ent_recall = []
    # Retrieve similar movies
    for i in indices[0]:
        ent_recall.append(ent_data[i])
    
    return ent_recall

@app.route('/lists', methods=['GET'])
def get_list():
    folder_name = request.args.get('folder_name', '')

    if folder_name:
        # sub_folder_name = f'{button_name}/' 
        files = get_files_from_s3(str(folder_name))
        # Extracting just the file names
        files = [os.path.basename(file) for file in files]
        return jsonify(files)
    else:
        return jsonify([]), 404

@app.route('/tri', methods=['POST'])
def triChat():
    print('tri - live')
    # Get text fields from form data
    username = request.form.get('username', 'noname')
    userId = request.form.get('userId', 'noid')
    gpt_prompt = request.form.get('prompt')
    feature = request.form.get('feature')
    selectedFile = request.form.get('activeFile')    
    print('file and feature: ',feature, selectedFile)
    
    S3_BUCKET = f'tri-cfo-uploads'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Access uploaded files
    if feature == 'Clip':
        uploaded_files = request.files.getlist('files')
        filename_list = []
        for file in uploaded_files:    
            try:
                filename_list.append(file.filename)
                if file.filename.endswith('.xlsx'):
                    # Handle Excel files
                    df = pd.read_excel(file)                    
                    print('xlsx uploaded successfully!')
                    gpt_prompt = f"{gpt_prompt} \nuploaded {file.filename} successfully."
                elif file.filename.endswith('.csv'):
                    # Handle CSV files
                    df = pd.read_csv(file)            
                    print('csv uploaded successfully!')
                    gpt_prompt = f"{gpt_prompt} \nuploaded {file.filename} successfully."
                else:
                    # Skip unsupported file types
                    df = pd.DataFrame()
                    print(f"no file loaded")        
                    continue

                # Reset the file's read pointer to the start
                file.seek(0)
                
                # Now upload to S3
                folder_name = userId
                upload_file_to_s3(file, folder_name, S3_BUCKET)
            except:
                print('file upload fail.')
                gpt_prompt = f"{gpt_prompt} \nfailed to upload file."

    if feature == 'Forecast':
        print('@ forecast')
        object_key = selectedFile

        you_are = "Hi, I'm Tri. I help you make more decicive action based on all context given"

        forecast_task = f"""data header = {df.head()}, relevant data types: {gpt_prompt}"""
        gpt_response = predGPT(forecast_task)

        folder_name = f"{userId}/ent_core"        

        new_data = {"timestamp": timestamp, "input": forecast_task, "output": gpt_response}
        append_to_json_in_s3(S3_BUCKET, folder_name, new_data)

        print("forecast_response:", gpt_response, "end")
        
        return jsonify(gpt_response)

    if feature == 'Analyze': 
        print('@ analyze')
        object_key = selectedFile

        you_are = "Hi, I'm Tri. I help you make more decicive action based on all context given"

        folder_name = userId
        df = read_csv_from_s3(S3_BUCKET, folder_name, object_key)       

        pandas_task = f"""data (df) = {df.head()} write python to execute based on this request: {gpt_prompt}"""
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

        print({"messages": [{"role": "user", "content": f"{pandas_task}"}, {"role": "assistant", "content": f"{python_code}"}]})
        
        gpt_input = f"{gpt_prompt} \ndata analysis: {str(python_result) + str(python_result_comment)}"
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
                    "content": gpt_input
                },
            ],
        )

        print('response:',response)
        gpt_response = response.choices[0].message['content'].strip()
        print(gpt_response)

        folder_name = f"{userId}/ent_core"        

        new_data = {"timestamp": timestamp, "input": gpt_input, "output": gpt_response}
        append_to_json_in_s3(S3_BUCKET, folder_name, new_data)

        return jsonify(gpt_response) #+ '\n\nResults:\n' + str(python_result) + str(python_result_comment))
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    object_key = f"ent_core/ent_core.json"

    you_are = f"timestamp: {timestamp}. Hi, I'm Tri. I help you make more decicive action based on all context given"

    folder_name = f"{userId}/ent_core"
    try:
        memory_recalled = read_json_from_s3(S3_BUCKET, folder_name, object_key, gpt_prompt)
    except:
        memory_recalled = ""

    gpt_input = f"from memory (may be related): {memory_recalled}. {gpt_prompt}"

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
                "content": gpt_input
            },
        ],
    )

    gpt_response = response.choices[0].message['content'].strip()
    print(gpt_response)

    folder_name = f"{userId}/ent_core"        

    new_data = {"timestamp": timestamp, "input": gpt_input, "output": gpt_response}
    append_to_json_in_s3(S3_BUCKET, folder_name, new_data)

    return jsonify(gpt_response)
### -tri- ###

### -upload- ###
def upload_file_to_s3(file, folder_name, bucket_name):
    try:
        filename = f"{folder_name}/{secure_filename(file.filename)}"
        s3.upload_fileobj(
            file,
            bucket_name,
            filename,
            ExtraArgs={
                "ContentType": file.content_type
            }
        )
        presigned_url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': filename}, ExpiresIn=3600)

        # Creating an additional "folder" inside the folder_name
        extra_folder_key = f"{folder_name}/ent_core/"
        s3.put_object(Bucket=bucket_name, Key=extra_folder_key)

        ent_core_key = f"{folder_name}/ent_core/ent_core.json"
        try:
            s3.head_object(Bucket=bucket_name, Key=ent_core_key)
            print("JSON file already exists in the extra_folder.")
        except ClientError as e:
            # If the file does not exist, create a new JSON file
            if e.response['Error']['Code'] == '404':
                json_data = {"timestamp": "init", "input": "core memory created", "output": "success"}
                s3.put_object(Bucket=bucket_name, Key=ent_core_key, Body=json.dumps(json_data))
                print("Created a new JSON file in the extra_folder.")

        return presigned_url
    except FileNotFoundError:
        return jsonify({"error": "The file was not found"}), 404
    except boto3.exceptions.NoCredentialsError:
        return jsonify({"error": "Credentials not available"}), 403
    except Exception as e:
        # Catch any other exception and return a meaningful error message
        return jsonify({"error": str(e)}), 500

        return presigned_url
    except FileNotFoundError:
        return jsonify({"error": "The file was not found"}), 404
    except NoCredentialsError:
        return jsonify({"error": "Credentials not available"}), 403
    except Exception as e:
        # Catch any other exception and return a meaningful error message
        return jsonify({"error": str(e)}), 500 

def append_to_json_in_s3(bucket_name, folder_name, new_data):
    s3 = boto3.client('s3')
    json_file_key = f"{folder_name}/ent_core/ent_core.json"

    try:
        # Try to fetch the existing JSON file
        response = s3.get_object(Bucket=bucket_name, Key=json_file_key)
        existing_data = json.loads(response['Body'].read().decode('utf-8'))

    except ClientError as e:
        # If the file does not exist, start with an empty list
        if e.response['Error']['Code'] == 'NoSuchKey':
            print("JSON file not found. Creating a new file.")
            existing_data = []
        else:
            raise

    # Append the new data
    existing_data.append(new_data)

    # Write the updated data back to the JSON file in S3
    updated_json_data = json.dumps(existing_data)
    s3.put_object(Bucket=bucket_name, Key=json_file_key, Body=updated_json_data)
    print("Data appended to the JSON file in S3.")       

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