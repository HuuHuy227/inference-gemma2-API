import requests

def test_chat_completions_api():
    url = "https://api.example.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY"
    }
    data = {
        "message": "Hello, how are you?",
        "max_completions": 5
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        completions = response.json()
        print("Completions:")
        for completion in completions:
            print(completion)
    else:
        print("Error:", response.status_code)

test_chat_completions_api()