import requests
import json

url = "http://localhost:8000/chat/stream"
payload = {"message": "Где находится офис и во сколько он открывается?"}

print("Ответ модели:\n")

with requests.post(url, json=payload, stream=True) as response:
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith("data: "):
                data_str = line_str[6:] # Отрезаем "data: "
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    # Печатаем текст без переноса строки и сразу сбрасываем буфер
                    print(data.get("text", ""), end="", flush=True) 
                except json.JSONDecodeError:
                    pass
print("\n\n--- Стриминг завершен ---")