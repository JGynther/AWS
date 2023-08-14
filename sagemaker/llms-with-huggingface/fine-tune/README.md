# Fine-tune a Huggingface model using Sagemaker

This example details running a training job on Sagemaker using the Huggingface integration in Sagemaker Python SDK and QLoRA.
The example tunes a [Falcon-7b base model](https://huggingface.co/tiiuae/falcon-7b) with the [CodeSearchNet dataset](https://huggingface.co/datasets/code_search_net). CodeSearchNet is a dataset of code-comment pairs. The goal is to make a generic base model generate code comments based on inputed code snippets.

> NOTE: this example assumes its running in Sagemaker (studio or notebook instance).

### Preparing data
The CodeSearchNet dataset is formatted into the following format:
```
### Input:
<code snippet>

### Response:
<code comment>
```
[prepare_dataset.py](prepare_dataset.py) details an example of how to format the dataset and upload it to S3 for our training job.

### Training job
Running a training job requires a few things
  - a training script ([train.py](scripts/train.py))
  - a [requirements.txt](scripts/requirements.txt)
  - a directory containing the above ([scripts](scripts))
  - a Huggingface estimator (detailed in the notebook)

> NOTE: The training could be run directly on a GPU-powered instance, but by using a training job you don't have to manage (and shutdown) the underlying instance(s) yourself.

The training script is the most complicated part of this process. It configures 4-bit quantization and LoRA. In this example hyperparemeters are hard coded into the script. Finally, the trained model artifact is uploaded to the configured S3 bucket, in this case Sagemaker's default bucket.

> Training script is based on [run_clm.py](https://github.com/huggingface/notebooks/blob/main/sagemaker/28_train_llms_with_qlora/scripts/run_clm.py)

### Deploying
[load-fine-tuned-local.ipynb](load-fine-tuned-local.ipynb) shows an example of loading a trained model for local inference. It assumes the `model.tar.gz` artifact has been downloaded to the local filesystem and unzipped, in this case to the `./model` directory.

Alternatively the model can be used with Sagemaker realtime inference:
```python
from sagemaker.huggingface.model import HuggingFaceModel

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   model_data="s3://path/to/your/model.tar.gz",    # path to your trained SageMaker model
   role=role,                                      # IAM role with permissions to create an endpoint
   transformers_version="4.8",                     # Transformers version used
   pytorch_version="2.0",                          # PyTorch version used
   py_version='py310',                             # Python version used
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
   initial_instance_count=1,
   instance_type="ml.g5.4xlarge"
)

# example request: you always need to define "inputs"
data = {
   "inputs": "Lorem ipsum dolor sit amet."
}

# request
predictor.predict(data)
```
Read more: [deploy with `model_data`](https://huggingface.co/docs/sagemaker/inference#deploy-with-modeldata) on Huggingface docs
#### Footnotes
Based on [Train LLMs using QLoRA on Amazon SageMaker](https://www.philschmid.de/sagemaker-falcon-qlora#4-deploy-fine-tuned-llm-on-amazon-sagemaker) and [Interactively fine-tune Falcon-40B and other LLMs on Amazon SageMaker Studio notebooks using QLoRA](https://aws.amazon.com/blogs/machine-learning/interactively-fine-tune-falcon-40b-and-other-llms-on-amazon-sagemaker-studio-notebooks-using-qlora/)
