import numpy as np
import random
from ecdsa import SECP256k1, ellipticcurve
import sys
import os
from numba import cuda
import numpy as np

# Целевой публичный ключ (пример)
TARGET_PUBLIC_KEY = "04d6597d465408e6e11264c116dd98b539740e802dc756d7eb88741696e20dfe7d3588695d2e7ad23cbf0aa056d42afada63036d66a1d9b97070dd6bc0c87ceb0d"
RESULT_FILE = "found_private_key.txt"

# Преобразование публичного ключа из строки в объект Point
def parse_public_key(pub_key_hex):
    x = int(pub_key_hex[2:66], 16)
    y = int(pub_key_hex[66:], 16)
    return ellipticcurve.Point(SECP256k1.curve, x, y)

target_pub_key = parse_public_key(TARGET_PUBLIC_KEY)

# Генерация случайного приватного ключа
def random_private_key():
    return random.randint(1, SECP256k1.order - 1)

# Оптимизированный алгоритм Полларда Ро для поиска приватного ключа на GPU
@cuda.jit
def pollards_rho_gpu(G_x, G_y, target_x, target_y, order, result, max_iterations):
    idx = cuda.grid(1)
    if idx >= max_iterations:
        return

    # Инициализация случайных значений для алгоритма
    x = idx % int(order)  # Использование индекса в качестве начального значения
    a = (idx * 2) % int(order)
    b = (idx * 3) % int(order)

    # Инициализация начальной точки
    X_x, X_y = (G_x * x + target_x * a) % order, (G_y * x + target_y * a) % order
    Y_x, Y_y = X_x, X_y

    for i in range(1, max_iterations):
        # Функция f(X) для итераций (упрощенно)
        X_x = (X_x * X_x) % order
        Y_x = (Y_x * Y_x) % order
        Y_x = (Y_x * Y_x) % order

        # Проверка на совпадение X и Y
        if X_x == Y_x and X_y == Y_y:
            r = (a - b) % order
            if r != 0:
                b_inv = cuda.local.array(1, dtype=np.int64)
                b_inv[0] = pow(b, -1, int(order))
                priv_key = (r * b_inv[0]) % order
                result[0] = priv_key
                return

# Основная функция запуска поиска приватного ключа
if __name__ == "__main__":
    generator = SECP256k1.generator
    max_iterations = 10**6
    num_threads = 256
    num_blocks = (max_iterations + (num_threads - 1)) // num_threads

    # Подготовка данных для передачи на GPU
    G_x, G_y = float(generator.x()), float(generator.y())
    target_x, target_y = float(target_pub_key.x()), float(target_pub_key.y())
    order = SECP256k1.order

    result = cuda.device_array(1, dtype=np.int64)

    print("Запуск поиска приватного ключа на GPU...")
    pollards_rho_gpu[num_blocks, num_threads](G_x, G_y, target_x, target_y, order, result, max_iterations)

    # Проверка результата
    private_key = result.copy_to_host()[0]
    if private_key:
        result_str = f"Найден приватный ключ: {private_key}\n"
        print(f"\n{result_str}")
        with open(RESULT_FILE, 'w') as f:
            f.write(result_str)
    else:
        print("\nПриватный ключ не найден. Попробуйте увеличить количество итераций.")