# **ORM_stack_a10_2_gpu_local_ELYZA-japanese_versions**

# **ORM Stack to deploy an A10.2 shape, one GPU and local vLLM ELYZA-Japanese**

## Installation
- **you can use Resource Manager from OCI console to upload the code from here**

## NOTE
- **the code deploys an A10.2 shape with one GPU Shape**
- **it requires a VCN and a subnet where the VM will be deployed**
- **it uses Oracle Linux image:**
- **for the image it will choose:**
```
sort_by                  = "TIMECREATED"
  sort_order               = "DESC"

  filter {
    name   = "launch_mode"
    values = ["NATIVE"]
  }
  filter {
    name = "display_name"
    values = ["\\w*GPU\\w*"]
    regex = true
  }
  ```
  **- it will add a freeform TAG : "GPU_TAG"= "A10-2"**
  **- the boot vol is 250 GB**
  **- the cloudinit will do all the steps needed to download and to start a vLLM Mistral model**
  ```
dnf install -y dnf-utils zip unzip
dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
dnf remove -y runc
dnf install -y docker-ce --nobest
systemctl enable docker.service
dnf install -y nvidia-container-toolkit
systemctl start docker.service
...
```
- **Few commands to check the progress and GPU resource utilization:**
```
monitor cloud init completion: tail -f /var/log/cloud-init-output.log
monitor multiple GPUs: nvidia-smi dmon -s mu -c 100 --id 0,1
```
**If needed:**
- **Start the model with this [ELYZA-japanese-Llama-2-7b or ELYZA-japanese-Llama-2-7b-instruct or ELYZA-japanese-Llama-2-7b-fast or ELYZA-japanese-Llama-2-7b-fast-instruct]:**
```
python -O -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port=8000 \
    --model "/home/opc/models/${MODEL}" \
    --tokenizer hf-internal-testing/llama-tokenizer \
    --enforce-eager \
    --max-num-seqs 1 \
    --tensor-parallel-size 2
    >> "${MODEL}.log" 2>&1 &
```
- **Test the model from CLI:**
```
curl -X POST "http://0.0.0.0:8000/generate" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Write a humorous limerick about the wonders of GPU computing.", "max_tokens": 64, "temperature": 0.7, "top_p": 0.9}'
	  
```
- **Or query with Jupyter notebook:**
```
import requests
import json

url = "http://0.0.0.0:8000/generate"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

data = {
    "prompt": "Write a short conclusion.",
    "max_tokens": 64,
    "temperature": 0.7,
    "top_p": 0.9
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    # Pretty print the response for better readability
    formatted_response = json.dumps(result, indent=4)
    print("Response:", formatted_response)
else:
    print("Request failed with status code:", response.status_code)
    print("Response:", response.text)
```
- **Or query with Gradio chat:**
```
import requests
import gradio as gr
import os

# Function to interact with the model via API
def interact_with_model(prompt):
    url = http://0.0.0.0:8000/generate
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    data = {
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.9
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        completion_text = result["text"][0].strip()  # Extract the generated text
        return completion_text
    else:
        return {"error": f"Request failed with status code {response.status_code}"}

# Retrieve the MODEL environment variable
model_name = os.getenv("MODEL")

# Example Gradio interface
iface = gr.Interface(
    fn=interact_with_model,
    inputs=gr.Textbox(lines=2, placeholder="Write a prompt..."),
    outputs=gr.Textbox(type="text", placeholder="Response..."),
    title=f"{model_name} Interface",  # Use model_name to dynamically set the title
    description=f"Interact with the {model_name} deployed locally via Gradio.",  # Use model_name to dynamically set the description
    live=True
)

# Launch the Gradio interface
iface.launch(share=True)
```
- **Start the model with Docker:**
**This needs further testing:**
```
docker run --gpus all \
-v "/home/opc/$MODEL/:/mnt/model" \
-p 8000:8000 \
--env "TRANSFORMERS_OFFLINE=1" \
--env "HF_DATASET_OFFLINE=1" \
--ipc=host vllm/vllm-openai:latest \
--model="/mnt/model" \
--tensor-parallel-size 2

docker run --gpus all \
-p 8000:8000 \
vllm/vllm-openai:latest \
--model="elyza/$MODEL" \
-e NCCL_DEBUG=INFO \
--tensor-parallel-size 2
```
- **This is not yet tested:**

- **Query the model working with Docker from CLI:(to be confirmed)**
```
curl -X POST "http://0.0.0.0:8000/generate" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain the importance of GPU computing in modern AI applications.", "max_tokens": 64, "temperature": 0.7, "top_p": 0.9}'
```
- **Query the model working with Docker from Jupyter notebook:(to be confirmed)**
```
import requests
import json

url = "http://0.0.0.0:8000/generate"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

data = {
    "prompt": "Write a short conclusion.",
    "max_tokens": 64,
    "temperature": 0.7,
    "top_p": 0.9
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    # Pretty print the response for better readability
    formatted_response = json.dumps(result, indent=4)
    print("Response:", formatted_response)
else:
    print("Request failed with status code:", response.status_code)
    print("Response:", response.text)
```
**Please keep in mind to allow firewall traffic at least for port 8888 used for Jupyter:**
```
sudo firewall-cmd --zone=public --permanent --add-port 8888/tcp
sudo firewall-cmd --reload
sudo firewall-cmd --list-all
```
**Please execute the following command to complete the autentication details for oci cli:**
```
oci setup config
```