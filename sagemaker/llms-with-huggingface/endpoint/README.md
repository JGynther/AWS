# Endpoint example
This example sets up a Sagemaker realtime inference endpoint using the `falcon-7b-instruct` model from Huggingface. In addition there is a simple chat interface built using Streamlit and Sagemaker Python SDK that shows how to call a Sagemaker endpoint iteratively for a snappy chat experience.

NOTE: this notebook and related Streamlit app presume they are running inside Sagemaker (studio or notebook instance). If you are running this locally, you will get an error regarding the Sagemaker session.

### Run the Streamlit app
```bash
pip install streamlit
streamlit run app.py
```
If you are running this in a notebook instance, you can access the Streamlit app at `https://YOUR-NOTEBOOK-HERE.notebook.AWS-REGION.sagemaker.aws/proxy/8501/`

NOTE: if you run into issues with XSRF try running the Streamlit instance without XSRF protection `streamlit run app.py --server.enableXsrfProtection=false`

