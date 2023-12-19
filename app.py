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
    return 'eb-live alpha tri v3.11'

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
                    df = pd.read_excel(file, engine='openpyxl')
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

        folder_name = userId
        df = read_csv_from_s3(S3_BUCKET, folder_name, object_key)

        forecast_task = f"""data header = {df.head()}, relevant data types: {gpt_prompt}"""
        gpt_response = predGPT(forecast_task)

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

        print('response:',response)
        gpt_response = response.choices[0].message['content'].strip()
        print(gpt_response)

        return jsonify(gpt_response) #+ '\n\nResults:\n' + str(python_result) + str(python_result_comment))

    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    you_are = f"timestamp: {timestamp}. Hi, I'm Tri. I help you make more decicive action based on all context given"

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
                "content": gpt_prompt
            },
        ],
    )

    gpt_response = response.choices[0].message['content'].strip()
    print(gpt_response)

    return jsonify(gpt_response)

    # Ensure the collection is retrieved or created
    # collection = Collection(userId)

    # filenames = ', '.join(filename_list)

    # if uploaded_files:
    #     gpt_prompt = gpt_prompt + 'uploaded files:'+ filenames
    #     context_layers = context_classifier(gpt_prompt + 'uploaded files:'+ filenames)
    # else:
    #     # context classifier
    #     context_layers = context_classifier(gpt_prompt)

    # try:
    #     # add context key and values
    #     for i in range(len(context_layers)):

    #         context_key = context_layers[i][0]
    #         context_value = context_layers[i][1]

    #         collection.add_to_dynamodb(
    #             documents=[context_value],
    #             metadatas=[context_key],
    #             timestamps=[timestamp],
    #         )        

    #         # multi-context search 
    #         search_results = []

    #         for i in range(len(context_layers)):

    #             context_key = context_layers[i][0]

    #             results = collection.query_from_dynamodb([gpt_prompt + context_key], n_results=3)

    #         search_results.append(results)
    #         print('search_results',search_results,'\n')
        
    # except:
    #     unique_strings = ['']

    # full retrieved context
    # Flatten the list of lists and remove duplicates using a set
    # unique_strings = set(item for sublist in search_results for item in sublist)

    # # Join the unique strings
    # joined_string = ' '.join(unique_strings)
    
    # # full context
    # if userId == 'Alpha':
    #     userId = "Steve"
    # full_context = f"{timestamp}. past chats wit me: {joined_string}. you: {gpt_prompt}."

    # collection.add_to_dynamodb(
    #     documents=[full_context],
    #     metadatas=[unique_strings],
    #     timestamps=[timestamp],
    # )        

    # you_are = "Hi, I'm Tri. I help you make more decicive action based on all context given"

    # # response
    # response = openai.ChatCompletion.create(
    #     model="ft:gpt-3.5-turbo-1106:triangleai::8U4LPTy8", #"gpt-3.5-turbo", #"gpt-4-1106-preview"
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": f"I am: {you_are}."
    #         },
    #         {
    #             "role": "user", 
    #             "content": full_context
    #         },
    #     ],
    # )

    # gpt_response = response.choices[0].message['content'].strip()
    # print(gpt_response)

    # # context classifier
    # context_layers_response = context_classifier(gpt_response)

    # try:
    #     # add context key and values
    #     for i in range(len(context_layers_response)):

    #         context_key = context_layers_response[i][0]
    #         context_value = context_layers_response[i][1]

    #         collection.add_to_dynamodb(
    #             documents=[context_value],
    #             metadatas=[context_layers_response],
    #             timestamps=[timestamp],
    #         ) 
    # except:
    #     print('context layers issue')

    # return jsonify(gpt_response)

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