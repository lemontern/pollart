import random
import os
from ecdsa import SECP256k1, SigningKey
from ecdsa.ellipticcurve import Point

# Целевой публичный адрес
TARGET_ADDRESS = "12ib7dApVFvg82TXKycWBNpN8kFyiAN1dr"
RESULT_FILE = "found_key.txt"
CHECKPOINT_FILE = "checkpoint.txt"

# Генерация случайных точек на кривой
def random_point():
    priv_key = random.randint(1, SECP256k1.order)
    pub_key = SECP256k1.generator * priv_key
    return priv_key, pub_key

# Алгоритм Pollard's Rho для поиска коллизий
# Цель состоит в нахождении двух разных приватных ключей, которые приводят к одной и той же точке публичного ключа.
def pollards_rho(curve, G, target_pub_key):
    # Инициализация точек
    x1, X1 = random_point()
    x2, X2 = x1, X1

    def f(X, x):
        if not isinstance(X, Point):
            raise ValueError("X должен быть объектом типа Point")
        if X.x() % 3 == 0:
            return (x + 1) % curve.order, X + G
        elif X.x() % 3 == 1:
            return (2 * x) % curve.order, 2 * X
        else:
            return (x - 1) % curve.order, X + (-G)

    while True:
        try:
            x1, X1 = f(X1, x1)
            x2, X2 = f(X2, x2)
            x2, X2 = f(X2, x2)
        except ValueError as e:
            print(f"Ошибка: {e}. Перезапуск...")
            x1, X1 = random_point()
            x2, X2 = x1, X1
            continue

        if X1 == X2:
            if x1 != x2:
                # Найдена коллизия
                try:
                    inv = pow(x1 - x2, -1, curve.order)
                except ValueError:
                    return None
                d = (x1 - x2) * inv % curve.order
                return d
            else:
                # Случай, когда x1 и x2 совпадают, необходимо перезапустить
                x1, X1 = random_point()
                x2, X2 = x1, X1

# Целевая точка публичного ключа, для которой необходимо найти приватный ключ.
last_checkpoint = None
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'r') as f:
        last_checkpoint = int(f.read().strip())

iteration = last_checkpoint if last_checkpoint else 0

while True:
    priv_key_target, pub_key_target = random_point()
    iteration += 1

    # Поиск приватного ключа с помощью алгоритма Pollard's Rho
    try:
        priv_key = pollards_rho(SECP256k1.curve, SECP256k1.generator, pub_key_target)
    except ValueError as e:
        print(f"Ошибка: {e}. Перезапуск...")
        continue

    if priv_key:
        result = f"Найден приватный ключ для адреса {TARGET_ADDRESS}: {priv_key}\n"
        print(result)
        with open(RESULT_FILE, 'w') as f:
            f.write(result)
        break
    else:
        print(f"Не удалось найти приватный ключ на итерации {iteration}. Продолжаем поиск...")
        with open(CHECKPOINT_FILE, 'w') as f:
            f.write(str(iteration))