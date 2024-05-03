from openai import OpenAI
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)

import argparse

def create_message(prompt):
    messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    return messages, query_dict

def get_response(client, messages, args):
    """
    Obtain response from GPT
    """
    SLEEP_TIME = 10
    success = False
    cnt = 0
    while not success:
        if cnt >= 50:
            rslt = "Error"
            break
        try:
            response = client.chat.completions.create(
                model=args.gpt_model,
                messages=messages,
                temperature=args.gpt_temperature,
                max_tokens=args.gpt_max_tokens,
                # frequency_penalty=frequency_penalty,
                # presence_penalty=presence_penalty,
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
    return rslt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--gpt_temperature", type=int, default=1)
    parser.add_argument("--gpt_max_tokens", type=int, default=256)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    prompt = "hello"
    client = OpenAI(
            api_key="YOUR_API_KEY",
        )

    messages = create_message(prompt)

    result = get_response(client, messages, args)
    print(result)