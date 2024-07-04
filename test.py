from chat_model import ChatModel

print("Loading model... ")
chat_model = ChatModel("Huy227/gemma2_vn")

def update_chat_history(history, content, max_length: int = 6):
    """
    Cập nhật lịch sử trò chuyện với phần tử mới và giữ tối đa max_length phần tử gần nhất.
    """
    history.append(content)
    if len(history) > max_length:
        history.pop(0)  # Loại bỏ phần tử cũ nhất để duy trì độ dài tối đa

chat_history = []

while True:
    question = input("Enter your question: ")
    update_chat_history(chat_history, {"role": "user", "content": question})
    if question == "exit":
        exit()
    reponse = chat_model.chat(
        prompt = question, 
        chat_history=chat_history,
        generate_config={
                "do_sample": True,
                "max_tokens": 2048,
                "top_p":0.95,
                "top_k":40,
                "temperature":0.5
    })

    reponse = reponse["choices"][0]["message"]["content"]
    update_chat_history(chat_history, {"role": "model", "content": reponse})
    # print(chat_history)
    print("REPONSE: ",reponse)
