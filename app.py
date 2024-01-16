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
    return 'eb-live alpha tri v3.22'

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
def codeGPT_run_python(prompt):    
    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-1106:triangleai::8XFC4wE8", #"ft:gpt-3.5-turbo-1106:triangleai::8U4LPTy8",
        messages=[
            {
                "role": "system",
                "content": "accelaration over efficiency. use pandas to analyze data based on request and return inline string to be run in exec(); example: exec('result = df.groupby('Gender')['Loan_Amount'].mean()')"
            },
            {
                "role": "user", 
                "content": f"{prompt}"
            },
        ],
    )

    gpt_response = response.choices[0].message['content'].strip()
    return gpt_response

def get_files_from_s3(folder):
    S3_BUCKET = 'tri-cfo-uploads'

    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder)
    files = [item['Key'] for item in response.get('Contents', [])]
    return files

def read_csv_from_s3(bucket_name, object_key):
    # Construct the full object key with folder and file name
    object_key = f"{object_key}"

    # Get the object from S3
    csv_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
    
    # Read the object's body into a DataFrame
    body = csv_obj['Body']
    csv_string = StringIO(body.read().decode('utf-8'))
    df = pd.read_csv(csv_string)
    
    return df

def read_json_from_s3(bucket_name, userId, prompt):
    # key drivers; personal, role, org, drivers, kpis, objectives etc.
    user_object_key = f"{userId}/ent_core/ent_core/user_core.json"

    # metadata - ent/org data, people, orgs, vocab, etc. 
    org_object_keys = f"{userId}/ent_core/ent_core/org_core.json"

    # metadata - added knowledge/context data, people, orgs, etc. 
    cfo_object_keys = f"{userId}/ent_core/ent_core/cfo_core.json"

    object_keys = [user_object_key, cfo_object_keys, org_object_keys]
    
    org_recall = []
    user_recall = []
    k_recall = []    

    n_k = len(object_keys)

    for key in object_keys:
        response = s3.get_object(Bucket=bucket_name, Key=key)        
            
        # Read the content of the file and parse it as JSON
        ent_data = json.loads(response['Body'].read())
        # start_time = time.time()
        # combined_texts = [f"timestamp: {data['timestamp']} input: {data['input']} output: {data['output']}" for data in ent_data]
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print('elapsed time for loop:',elapsed_time)

        # start_time = time.time()
        combined_texts = list(map(lambda data: f"timestamp: {data['timestamp']} input: {data['input']} output: {data['output']}", ent_data))
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print('elapsed time for list:',elapsed_time)

        # Vectorizing text data using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(combined_texts)

        # Using KNN to find similar movies
        knn = NearestNeighbors(n_neighbors=n_k, metric='cosine')
        knn.fit(tfidf_matrix)

        query_tfidf = vectorizer.transform([prompt])
        
        # Find similar movies
        distances, indices = knn.kneighbors(query_tfidf, n_neighbors=n_k)

        # Combine indices and distances, and sort by distances in descending order
        sorted_indices_distances = sorted(zip(indices[0], distances[0]), key=lambda x: x[1], reverse=True)

        # Retrieve and print similar movies along with their distances
        for i, distance in sorted_indices_distances:
            if "org_core.json" in key:                
                print(f"Ent Memory: {ent_data[i]}, Distance: {distance}")
                org_recall.append(ent_data[i])
            if "user_core.json" in key:
                print(f"User Memory: {ent_data[i]}, Distance: {distance}")
                user_recall.append(ent_data[i])
            if "cfo_core.json" in key:
                print(f"K Memory: {ent_data[i]}, Distance: {distance}")
                k_recall.append(ent_data[i])
                
    org_recall = ' '.join([str(dictionary) for dictionary in org_recall])
    user_recall = ' '.join([str(dictionary) for dictionary in user_recall])
    k_recall = ' '.join([str(dictionary) for dictionary in k_recall])

    return org_recall, user_recall, k_recall

@app.route('/lists', methods=['GET'])
def get_list():
    userId = request.args.get('userId', '')
    directory = request.args.get('directory', '')
    print("/lists - directory:", directory)
    
    if userId:
        # Ensure the directory path ends with a '/'
        if directory and not directory.endswith('/'):
            directory += '/'

        items = get_files_from_s3(f"{userId}/{directory}")
        # Calculating the depth of the specified directory
        directory_depth = directory.count('/')

        # Separating files and folders
        files = []
        folders = []
        for item in items:
            # Exclude the directory itself from the list
            if item == f"{userId}/{directory}":
                continue

            # Removing the user ID and slash from the item path for proper comparison
            relative_item_path = item[len(f"{userId}/"):]

            # Calculate the depth of each item
            item_depth = relative_item_path.count('/')

            # Check if item is directly in the directory
            if item_depth == directory_depth or (relative_item_path.endswith('/') and item_depth == directory_depth + 1):
                if relative_item_path.endswith('/'):
                    folders.append(os.path.basename(relative_item_path[:-1]) + '/')
                else:
                    files.append(os.path.basename(relative_item_path))

        # Sorting files and folders alphabetically
        folders.sort()
        files.sort()

        # Combining folders and files, with folders first
        all_items = folders + files

        return jsonify(all_items)
    else:
        return jsonify([]), 404

@app.route('/upload_file', methods=['POST'])
def upload_file():
    files = request.files.getlist('files')
    userId = request.form.get('userId', '')
    directory = request.form.get('directory', '')
    bucket_name = 'tri-cfo-uploads' 

    for file in files:
        if file:
            filename = secure_filename(file.filename)
            object_key = f'{userId}/{directory}{filename}'
            print(object_key)
            try:
                s3.upload_fileobj(file, bucket_name, object_key)
                print(f"Uploaded {filename} to S3 bucket {bucket_name} in directory {directory}")
            except Exception as e:
                print(f"Error uploading {filename}: {e}")
                return jsonify(f"Error uploading {filename}"), 500

    print(f"User ID: {userId}, Directory: {directory}")
    return jsonify('Files uploaded successfully')

@app.route('/add_file', methods=['POST'])
def add_file():
    # Access other form data
    userId = request.form.get('userId', '')
    directory = request.form.get('directory', '')
    filename = request.form.get('filename', '')

    bucket_name = 'tri-cfo-uploads'

    # Create an empty CSV file-like object
    csv_buffer = StringIO()
    csv_buffer.write('') 

    # Make sure we are at the start of the StringIO buffer
    csv_buffer.seek(0)

    # Create the full file path
    key = f"{userId}/{directory}{filename}"
    print(key)

    # Upload the empty CSV file to S3
    s3.put_object(Bucket=bucket_name, Key=key, Body=csv_buffer.getvalue())

    return jsonify('Empty CSV file uploaded successfully')

@app.route('/add_file', methods=['POST'])
def add_file():
    # Access other form data
    userId = request.form.get('userId', '')
    directory = request.form.get('directory', '')
    filename = request.form.get('filename', '')

@app.route('/add_folder', methods=['POST'])
def add_folder():
    data = request.get_json()  # Get JSON data from the request body
    print('/add_folder data POST:',data)

    userId = data.get('userId', '')
    folder_name = data.get('foldername', '')
    
    directory = data.get('directory', '')
    print("/add_folder - directory:",directory)

    if not folder_name.endswith('/'):
        folder_name += '/'  # Ensure the folder name ends with a '/'

    s3_client = boto3.client('s3')
    S3_BUCKET = 'tri-cfo-uploads'  # Replace with your bucket name

    # The key is the full path within the bucket, including the folder name
    key = f"{userId}/{directory}{folder_name}"

    # Create an empty object to represent the folder
    s3_client.put_object(Bucket=S3_BUCKET, Key=key)

    return jsonify('Folder created successfully')

@app.route('/delete_file', methods=['POST'])
def delete_file():
    data = request.get_json()
    userId = data.get('userId', '')
    filename = data.get('filename', '')
    directory = data.get('directory', '')

    S3_BUCKET = 'tri-cfo-uploads'
    object_key = f'{userId}/{directory}{filename}'

    try:
        if filename.endswith('/'):  # If it's a folder
            # List and delete all objects in the folder
            objects_to_delete = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=object_key)
            if 'Contents' in objects_to_delete:
                delete_keys = {'Objects': [{'Key': obj['Key']} for obj in objects_to_delete['Contents']]}
                s3.delete_objects(Bucket=S3_BUCKET, Delete=delete_keys)

            # Additionally, delete the placeholder object for the folder
            s3.delete_object(Bucket=S3_BUCKET, Key=object_key)

            response_per_user = f'Folder "{filename}" has been deleted successfully.'
        else:  # If it's a file
            s3.delete_object(Bucket=S3_BUCKET, Key=object_key)
            response_per_user = f'File "{filename}" has been deleted successfully.'

        return jsonify({'message': response_per_user})

    except ClientError as e:
        print(str(e))
        return jsonify({'message': 'File or folder deletion failed.'}), 500

@app.route('/get_data', methods=['POST'])
def get_data():
    data = request.get_json()
    userId = data.get('userId', '')
    filename = data.get('filename', '')
    directory = data.get('directory', '')

    # Define your bucket name here
    S3_BUCKET = f'tri-cfo-uploads'
    object_key = f'{userId}/{directory}{filename}'
    print('inside /get_data object_key:',object_key)
    
    try:
        df = read_csv_from_s3(S3_BUCKET, object_key)
        # Convert DataFrame to JSON
        # json_str = df.to_json(orient='records')
        json_str = df.head(7).to_json(orient='records')
        
        # Return JSON response
        return jsonify(json_str)

    except Exception as e:        
        return jsonify(pd.DataFrame().to_json(orient='records'))

@app.route('/tri', methods=['POST'])
def triChat():
    # print('tri - live')
    # Get text fields from form data
    username = request.form.get('username', 'noname')
    userId = request.form.get('userId', 'noid')
    gpt_prompt = request.form.get('prompt')
    feature = request.form.get('feature')
    selectedFile = request.form.get('activeFile')    
    directory = request.form.get('directory', '')
    print("/tri directory:",directory)    
    
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
                    gpt_prompt = f"{gpt_prompt} \nuploaded {file.filename} successfully."
                elif file.filename.endswith('.csv'):                    
                    gpt_prompt = f"{gpt_prompt} \nuploaded {file.filename} successfully."
                elif file.filename.endswith('.pdf') or file.filename.endswith('.docx'):
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

    if feature == 'Analysis': 
        print('@ Analysis')
        object_key = selectedFile

        you_are = "Hi, you are Tri. You help me analyze data and explain your analysis clearly and consicely while providing requested data and other details."
        
        S3_BUCKET = f'tri-cfo-uploads'
        object_key = f'{userId}/data/{selectedFile}'

        df = read_csv_from_s3(S3_BUCKET, object_key) 

        for attempt in range(3):
            try:           
                cols = df.columns.tolist()
                python_code = codeGPT_run_python(f"{gpt_prompt}:: use df, columns: {cols} response must have must respond with a final result variable")
                # print(python_code)

                local_vars = {'df': df}
                exec(python_code, globals(), local_vars)

                # If execution is successful, break out of the loop
                break
            except Exception as e:
                # Handle the exception (e.g., log it)
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt == 2:
                    # Handle the failure after the final attempt
                    print("Failed after 3 attempts")
                    return jsonify('can you rephrase that?')

        # Retrieve the result from local_vars
        python_result = local_vars['result']

        try:
            print("python_result:", str(python_result))
        except:
            print("python_result:", str(python_result))

        # print({"messages": [{"role": "user", "content": f"{pandas_task}"}, {"role": "assistant", "content": f"{python_code}"}]})
        
        gpt_input = f"{gpt_prompt} \ndata analysis: {str(python_result)}"
        # response
        response = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-1106:triangleai::8U4LPTy8", #"gpt-3.5-turbo", #"gpt-4-1106-preview"
            messages=[
                {
                    "role": "system",
                    "content": f"You are: {you_are}."
                },
                {
                    "role": "user", 
                    "content": gpt_input
                },
            ],
        )

        # print('response:',response)
        gpt_response = response.choices[0].message['content'].strip()
        # print(gpt_response)
   
        new_data = {"timestamp": timestamp, "input": gpt_input, "output": gpt_response}
        append_to_json_in_s3(S3_BUCKET, userId, new_data)

        return jsonify(gpt_response) #+ '\n\nResults:\n' + str(python_result) + str(python_result_comment))
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    you_are = "Hi, you are Tri. You help me make better decicive action. Time is relevant. Never reveal 'core' value. Match my tone. Ask when unsure."

    try:
        org_recall, user_recall, k_recall = read_json_from_s3(S3_BUCKET, userId, gpt_prompt)
    except:
        org_recall, user_recall, k_recall = "", "", ""

    gpt_input_w_mem = f"current timestamp: {timestamp}, my prompt: {gpt_prompt}. our past conversations: {user_recall}, my organization info up to this point: {org_recall}, Tri knowledge: {k_recall}. Tri: "

    # response
    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-1106:triangleai::8U4LPTy8", #"gpt-3.5-turbo" "gpt-4-1106-preview"
        messages=[
            {
                "role": "system",
                "content": f"You are: {you_are}."
            },
            {
                "role": "user", 
                "content": 'user: who are you?'
            },
            {
                "role": "assistant", 
                "content": "user: My name is Tri, and I'm an AI"
            },
            {
                "role": "user", 
                "content": 'user:' + gpt_input_w_mem
            },
        ],
    )

    gpt_response = response.choices[0].message['content'].strip()
    # print(gpt_response)

    new_data = {"timestamp": timestamp, "input": gpt_prompt, "output": gpt_response}
    append_to_json_in_s3(S3_BUCKET, userId, new_data)

    return jsonify(gpt_response)
### -tri- ###

### -upload- ###
def upload_file_to_s3(file, folder_name, bucket_name):
    try:
        filename = f"{folder_name}/data/{secure_filename(file.filename)}"
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

        ent_core_key = f"{folder_name}/ent_core/ent_core/org_core.json"
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

def append_to_json_in_s3(bucket_name, userId, new_data):
    # s3 = boto3.client('s3')
    json_file_key = f"{userId}/ent_core/ent_core/user_core.json"

    try:
        # Try to fetch the existing JSON file
        response = s3.get_object(Bucket=bucket_name, Key=json_file_key)
        existing_data = json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        # Check if the error is because the file does not exist
        if e.response['Error']['Code'] == 'NoSuchKey':
            print("JSON file not found. Creating a new file.")
            existing_data = []
        else:
            # Re-raise the error if it's not a 'NoSuchKey' error
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