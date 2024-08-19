# Continua Model Setup and FastAPI Application

This guide will walk you through the process of pulling a model from an S3 bucket, setting it up in Ollama, installing Poetry, creating a virtual environment, and running a FastAPI application.

If this were being deployed on-device, it would have to be implemented differently, but the core logic would be the same.

I have finetuned a llama 3.1 8b model that we will pull in to use as the local model.

Alternatively, you can use the standard llama3.1 in ollama but it would yield less accurate results.

Further work on this problem would be to add unit tests, add authN/Z, and add input validation and error handling as well as some better logging and observability/tracing.

We also would probably want to dockerize the service.

## Prerequisites

- **AWS CLI**: Ensure you have the AWS CLI installed and configured.
- **Ollama**: Ensure Ollama is installed and configured on your machine.
- **Python 3.10+**: Python should be installed on your machine.

## Steps

### 1. Pulling the Model from S3

1. **Set up AWS Credentials**:
   Ensure your AWS credentials are set up correctly as environment variables:
   
   ```sh
   export AWS_ACCESS_KEY_ID=your_access_key_id
   export AWS_SECRET_ACCESS_KEY=your_secret_access_key
   export AWS_SESSION_TOKEN=your_session_token_key
   export AWS_DEFAULT_REGION=your_aws_region
   ```

   or make sure your local awscli env is configured so the default credentials provider can reference it.

   The S3 bucket is public read, but your credentials will need permissions on `bedrock:InvokeModel` and Claude Sonnet 3.5 will need to be enabled in your account.

   You can copy `.env.template` into `.env` and set your env vars how you wish.

2. **Download the finetuned model artifacts**:

   This will be quite large, if you just want to test the service, just use llama3.1, in reality we would play around more with the quantization and lora adapter sizes to meet our requirements.
   
   ```sh
   mkdir model
   aws s3 sync s3://bilyana-continua/model model
   ```

3. **Setup the model in ollama**:

   ```sh
   ollama create continua -f model/Modelfile
   ollama run continua
   ```

   You will need to change the model name env var `LOCAL_MODEL_NAME` in your `.env` to use this model, or by default it will use `llama3.1`.

4. **Install poetry and dependencies**:
   ```sh
   python3 -m pip install --upgrade pip && \
   python3 -m pip install virtualenv && \
   python3 -m venv .venv && \
   source .venv/bin/activate && \
   pip install poetry && \
   poetry env use $(which python) && \
   poetry install --no-root
   ```

   or

   ```sh
   ./setup_env.sh
   ```

5. **Run the service**:
   Replace the port with which
   ```sh
   poetry run uvicorn app.main:app --port 8080
   ```

6. **Make a request**:
   You can use whatever you want, easiest is probably in the browser at `http://localhost:8080/docs` (or another available port).

   Or you can always use python `requests`, or `curl`

   ```python
   import requests

   url = 'http://127.0.0.1:8080/api/inference/generate-plan'
   headers = {
      'accept': 'application/json',
      'Content-Type': 'application/json'
   }
   data = {
      "utterance": "schedule me a lunch with Jennifer and her team tomorrow at noon"
   }

   response = requests.post(url, headers=headers, json=data)

   print(response.status_code)
   print(response.json())

   ```

   ```sh
   curl -X 'POST' \
   'http://127.0.0.1:8080/api/inference/generate-plan' \
   -H 'accept: application/json' \
   -H 'Content-Type: application/json' \
   -d '{
   "utterance": "schedule me a lunch with Jennifer and her team tomorrow at noon"
      }'
   ```

   ## Delegation and Inference

   For delegation, we ask the small model to give us a confidence score (not optimal, but valid, see [this paper](https://arxiv.org/html/2406.03441v1)), and we keep a sliding window of these scores in a bounded list. 

   We take the p20 confidence (to adjust for the small model's biases for score ranges) and send the lowest confidence 20% to Claude Sonnet 3.5 via Amazon Bedrock if the delegation ration is < 20%.

   We also take a running count to calculate the ratio up to a maximum bound before it resets. One caveat is that when `n` is small in the counter, the ratios will be skewed, and the first call will always go to the large model.

   We finetuned the small model and also provide few-shot examples to both models.

   In the future, we would probably want to train some lightweight classifier to decide if something is complex enough to be delegated, as well as use the other information in the data such as tags, turn and canonical tokens.

   We could index the training data utterances with their plans to allow us to perform RAG to retrieve the most relevant examples for few-shot prompts.
