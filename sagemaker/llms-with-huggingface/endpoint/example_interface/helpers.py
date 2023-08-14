import sagemaker
from sagemaker.predictor import Predictor
import json

sess = sagemaker.Session()
role = sagemaker.get_execution_role()


ENDPOINT = "YOUR_ENDPOINT_NAME_HERE"
TOKENS_PER_INVOKE = 5
MAX_NEW_TOKENS = 1024


predictor = Predictor(ENDPOINT)


def predict(prompt: str):
    payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.8,
            "max_new_tokens": TOKENS_PER_INVOKE,
            "repetition_penalty": 1.03,
            "stop": ["\nUser:", "<|endoftext|>", "</s>"],
        },
    }

    response = predictor.predict(
        data=json.dumps(payload), initial_args={"ContentType": "application/json"}
    )
    response = json.loads(response)

    return response[0]["generated_text"][len(prompt) :]


class StreamHandler:
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def _update(self, token: str) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")  # Fake a cursor to mimic writing

    def _replace(self, text: str) -> None:
        self.container.markdown(self.text)

    def stream_iterate_tokens(self, prompt: str) -> str:
        for _ in range(0, MAX_NEW_TOKENS, TOKENS_PER_INVOKE):
            response = predict(prompt)

            if response == "":
                break

            self._update(response)
            prompt += response

        self._replace(self.text)
        return self.text
