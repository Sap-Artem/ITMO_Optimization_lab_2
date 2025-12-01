# ITMO_Optimization_lab_2
Автор: Сапожников А.А. МетОпт 1.1
### Пример
```text
Найти минимум функции: 10 + (x**2 - 10*cos(2*pi*x))
Период: [-5.12, 5.12]
```
### Код
#### Оценка константы Липшица 
```python
def estimate_lipschitz(f, a, b, samples=1000):
    """
    Приблизительно оценивает константу Липшица по конечным разностям.
    """
    X = np.linspace(a, b, samples)
    Y = f(X)
    slopes = np.abs(np.diff(Y) / np.diff(X))
    # Берём 98‑й процентиль, чтобы избежать выбросов
    est = np.percentile(slopes, 98)
    return float(est if est > 0 else 1.0)
```
#### Алгоритм Пиявского-Шуберта
```python
def piyavskii_minimize(f, a, b, L, eps=1e-3, max_iter=10000):
    """
    Реализация алгоритма глобальной оптимизации Пиявского–Шуберта.

    f        — функция
    a, b     — границы интервала
    L        — константа Липшица (или её верхняя оценка)
    eps      — критерий остановки (f_best - LB_min <= eps)
    max_iter — ограничение итераций

    Возвращает словарь с результатами и историей вычислений.
    """

    t0 = time.time()

    # Начальные точки — концы интервала
    xs = [a, b]
    fs = [float(f(a)), float(f(b))]

    # Вставка новой точки в отсортированный список
    def insert_point(x_new, f_new):
        for i, xi in enumerate(xs):
            if x_new < xi:
                xs.insert(i, x_new)
                fs.insert(i, f_new)
                return
        xs.append(x_new)
        fs.append(f_new)

    it = 0
    history = []

    while it < max_iter:
        it += 1

        candidates = []

        for i in range(len(xs) - 1):
            x_i, x_j = xs[i], xs[i+1]
            f_i, f_j = fs[i], fs[i+1]

            # Формула нахождения точки пересечения "конусов"
            x_c = 0.5 * (x_i + x_j) + (f_j - f_i) / (2 * L)

            # Коррекция: точка должна находиться внутри интервала
            if x_c <= x_i or x_c >= x_j:
                x_c = 0.5 * (x_i + x_j)

            # Нижняя граница на интервале
            LB_value = 0.5 * (f_i + f_j - L * (x_j - x_i))

            candidates.append((x_c, LB_value, i))

        # Выбираем интервал с минимальной нижней границей
        x_star, LB_min, idx = min(candidates, key=lambda t: t[1])

        # Лучшая найденная точка
        best_idx = int(np.argmin(fs))
        x_best = xs[best_idx]
        f_best = fs[best_idx]

        # Сохраняем историю
        history.append((xs.copy(), fs.copy(), x_best, f_best, LB_min))

        # Критерий остановки
        if (f_best - LB_min) <= eps:
            break

        # Вычисляем f(x*) и добавляем новую точку
        f_star = float(f(x_star))
        insert_point(x_star, f_star)

    total_time = time.time() - t0

    best_idx = int(np.argmin(fs))
    return {
        "x_best": xs[best_idx],
        "f_best": fs[best_idx],
        "iterations": it,
        "time": total_time,
        "xs": xs,
        "fs": fs,
        "history": history,
        "L": L,
        "eps": eps,
    }
```
#### Визуализация
```python
def plot_search(f, a, b, result, filename_prefix="result"):
    """
    Рисует:
    • исходную функцию
    • ломаную по выборке
    • нижнюю оценку (максимум из конусов)
    • найденный минимум
    """

    xs = np.array(result["xs"])
    fs = np.array(result["fs"])
    L = result["L"]

    # Плотная сетка для функции
    X = np.linspace(a, b, 2000)
    Y = f(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(X, Y, label="f(x)", linewidth=1.5)

    # Точки выборки
    ax.plot(xs, fs, "o-", label="Точки выборки", markersize=5)

    # Нижняя огибающая
    cones = np.array([fs[i] - L * np.abs(X - xs[i]) for i in range(len(xs))])
    LB = cones.max(axis=0)
    ax.plot(X, LB, "--", label="Нижняя оценка", linewidth=1.2)

    # Найденный минимум
    ax.plot(result["x_best"], result["f_best"], "r*", markersize=12, label="Минимум")

    # Подпись результата
    info = (
        f"x* = {result['x_best']:.6g}\n"
        f"f(x*) = {result['f_best']:.6g}\n"
        f"итераций: {result['iterations']}\n"
        f"время: {result['time']:.4f} c"
    )
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.85),
    )

    ax.set_title(f"Метод Пиявского–Шуберта (L={L}, eps={result['eps']})")
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    fig.tight_layout()
    plt.show()
```
Оценка L: 69.16937550585916
x_best: 0.0
f_best: 0.0
iterations: 168,
time: 0.007007598876953125,
L: 69.16937550585916,
eps: 0.01
```python
f = parse_function('10 + (x**2 - 10*cos(2*pi*x))')
a, b = -5.12, 5.12

L = estimate_lipschitz(f, a, b, samples=2000)
print("Оценка L:", L)

result = piyavskii_minimize(f, a, b, L, eps=0.01)
print(result)
```
#### Результат
<img width="1223" height="723" alt="image" src="https://github.com/user-attachments/assets/4dfdf2e5-ce1b-4b94-89a9-21e65a7db7a9" />
