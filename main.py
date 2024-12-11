from pathlib import Path
import numpy as np

from utils import fix_zero_rounding_errors as fzre

# Конструирование пути к папке матриц
current_path = Path(".").absolute()
data_path = current_path / "data"
a_path = data_path / "A.txt"
b_path = data_path / "B.txt"


def p_indices(x: np.ndarray, b: np.ndarray, f: np.ndarray) -> tuple[list, list]:
    """Возвращает индексы столбцов bj и fi, включенных в множество p(x)"""
    # np.isclose используется из-за проблем с точностью np.float64
    f_indices = [i for i, _f in enumerate(f)   if np.isclose(_f @ x.T, 0)]
    b_indices = [i for i, _b in enumerate(b.T) if np.isclose(_b @ x.T, 1)]
    return b_indices, f_indices


def q_indices(y: np.ndarray, a: np.ndarray, e: np.ndarray) -> tuple[list, list]:
    """Возвращает индексы столбцов ai и ej, включенных в множество q(y)"""
    # np.isclose используется из-за проблем с точностью np.float64
    e_indices = [i for i, _e in enumerate(e.T) if np.isclose(_e @ y.T, 0)]
    a_indices = [i for i, _a in enumerate(a)   if np.isclose(_a @ y.T, 1)]
    return a_indices, e_indices


def is_nash_equilibrium(
    x: np.ndarray, y: np.ndarray,
    a: np.ndarray, b: np.ndarray,
    e: np.ndarray, f: np.ndarray
) -> bool:
    """Является ли пара стратегий (x, y) равновесной по Нэшу?"""
    a_indices, e_indices = q_indices(y, a, e)
    b_indices, f_indices = p_indices(x, b, f)

    m, n = a.shape
    e_part = set(r for r in range(n) if r in e_indices or r in b_indices)
    f_part = set(s for s in range(m) if s in f_indices or s in a_indices)
    return len(e_part) == n and len(f_part) == m


def get_basis_transformation_matrix(old_basis, pivot_row, new_row) -> np.ndarray:
    """Возвращает матрицу перехода к новому базису"""
    t = old_basis.copy()
    e = old_basis[pivot_row]

    m, n = old_basis.shape
    for j in range(n):
        s = sum(new_row[k] * old_basis[k, j] for k in range(m) if k != pivot_row)
        t[pivot_row, j] = (e[j] - s) / new_row[pivot_row]
    return t


def find_equilibrium_strategy(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = a.shape
    m, n = shape

    e, f = np.identity(n), np.identity(m)

    # 3. Формирование таблицы А0*
    alpha, q = a.copy(), e.copy()

    for a_start_index in range(n):
        # 4. Начальное значение y0
        y_value = np.min(a[:, a_start_index])
        pivot_row = np.argmin(a[:, a_start_index])

        y0 = np.zeros(n, dtype=np.float64)
        y0[a_start_index] = 1 / y_value
        y = y0[:]

        # 5. Формирование таблицы B0*
        beta, p = b.copy(), f.copy()

        # 6. Начальное значение x0
        x_value = np.min(b[pivot_row])
        pivot_column = np.argmin(b[pivot_row])

        x0 = np.zeros(m, dtype=np.float64)
        x0[pivot_row] = 1 / x_value
        x = x0[:]

        # 7. Проверка условий равновесия
        if is_nash_equilibrium(x, y, a, b, e, f):
            return x, y

        # 8. Замена базиса
        a_transform = get_basis_transformation_matrix(e, a_start_index, alpha[pivot_row])
        alpha, q = fzre(alpha @ a_transform), fzre(q @ a_transform)

        b_transform = get_basis_transformation_matrix(f, pivot_row, b.T[pivot_column])
        beta, p = fzre((b.T @ b_transform).T), fzre(f @ b_transform)

        # 9. Вычисление ξ и η
        xi = (a @ y.T)
        eta = (b.T @ x.T).T

        # 10. Вычисление λ и μ
        lambda_ = np.zeros(n)
        for j in range(n):
            lambda_negative = min([
                *[-(xi[k] - 1) / alpha[k, j] for k in range(m) if alpha[k, j] < 0],
                *[-(y0[r]) / q[j, r]         for r in range(n) if q[j, r] < 0]
            ], default=0)
            lambda_positive = max([
                *[-(xi[s] - 1) / alpha[s, j] for s in range(m) if alpha[s, j] > 0],
                *[-(y0[t]) / q[j, t]         for t in range(n) if q[j, t] > 0]
            ], default=0)
            lambda_[j] = lambda_negative if lambda_negative != 0 else lambda_positive

        mu = np.zeros(m)
        for i in range(m):
            mu_negative = min([
                *[-(eta[r] - 1) / beta[i, r] for r in range(n) if beta[i, r] < 0],
                *[-(x0[k]) / p[k, i]         for k in range(m) if p[k, i] < 0]
            ], default=0)
            mu_positive = max([
                *[-(eta[t] - 1) / beta[i, t] for t in range(n) if beta[i, t] > 0],
                *[-(x0[s]) / p[s, i]         for s in range(m) if p[s, i] > 0]
            ], default=0)
            mu[i] = mu_negative if mu_negative != 0 else mu_positive

        # 11. Определение оптимальной стратегий
        for i in range(m):
            x = x0 + mu[i] * p[:, i]
            for j in range(n):
                y = y0 + lambda_[j] * q[:, j]
                if is_nash_equilibrium(x, y, a, b, e, f):
                    return x, y

    raise Exception("Не смогли найти равновесие в смешанных стратегиях")


def main():
    # Ввод данных
    a_initial: np.ndarray = np.loadtxt(a_path, delimiter=" ")
    b_initial: np.ndarray = np.loadtxt(b_path, delimiter=" ")
    if a_initial.shape != b_initial.shape:
        raise ValueError(f"Матрицы A {a_initial.shape} и B {b_initial.shape} должны быть одной размерности")

    shape = a_initial.shape
    m, n = shape

    # 1. Определение числа d
    d = np.max([a_initial, b_initial]) + 1

    # 2. Формирование матриц A1 и B1
    a = d * np.ones(shape) - a_initial
    b = d * np.ones(shape) - b_initial

    # Ищем равновесные стратегии x~ и y~
    x, y = find_equilibrium_strategy(a, b)

    # Нормализация стратегий
    x_star, y_star = x / np.sum(x), y / np.sum(y)

    # Цена игры
    hb = d - 1 / np.sum(x)
    ha = d - 1 / np.sum(y)

    print("Оптимальная стратегия x*:", x_star)
    print("Оптимальная стратегия y*:", y_star)
    print("Цена игры для игрока A:", ha)
    print("Цена игры для игрока B:", hb)


if __name__ == "__main__":
    main()
