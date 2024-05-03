from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)
import time
from models.key import _API_KEY

class ChatGPT:
    def __init__(self, model_name, args):
        self.client = OpenAI(
            api_key=_API_KEY,
        )
        self.model = model_name
        self.args = args
    
    def process_responses(self, responses, prompts):
        outputs = []
        for prompt, response in zip(prompts, responses):
            response = response.replace(prompt, "")
            outputs.append(response.strip())
        return outputs

    def generate(self, prompts):
        SLEEP_TIME = 10
        responses = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}] 
            success = False
            cnt = 0
            while not success:
                if cnt >= 50:
                    rslt = "Error"
                    break
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.args.temperature,
                        max_tokens=self.args.max_new_tokens,
                    )
                    rslt = response.choices[0].message.content
                    success = True
                except RateLimitError as e:
                    print(f"sleep {SLEEP_TIME} seconds for rate limit error")
                    time.sleep(SLEEP_TIME)
                except APITimeoutError as e:
                    print(f"sleep {SLEEP_TIME} seconds for time out error")
                    time.sleep(SLEEP_TIME)
                except APIConnectionError as e:
                    print(f"sleep {SLEEP_TIME} seconds for api connection error")
                    time.sleep(SLEEP_TIME)
                except APIError as e:
                    print(f"sleep {SLEEP_TIME} seconds for api error")
                    time.sleep(SLEEP_TIME)
                except Exception as e:
                    print(e)
                    success = True
                    rslt = "Error"
                cnt += 1
            responses.append(rslt)
        return responses