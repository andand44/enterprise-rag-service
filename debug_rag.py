import vector_db
import time

print("Начинаем поиск в Qdrant с переранжированием...")
start = time.time()

# Вызываем только поиск, без Ollama
result = vector_db.search_docs("Где находится офис?")

end = time.time()
print(f"Поиск занял: {end - start:.2f} секунд")
print("\nНайденный контекст:")
print(result)