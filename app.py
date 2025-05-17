import streamlit as st
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========== Hàm giải Simplex ========== #
def simplex_full_phase(c, A, b, signs):
    m, n = len(A), len(A[0])
    tableau = []
    basis = []
    artificial = []

    # Xử lý từng ràng buộc
    for i in range(m):
        row = A[i][:]
        slack = [0] * m
        artificial_var = [0] * m

        if signs[i] == "<=":
            slack[i] = 1
            row += slack + artificial_var + [b[i]]
            basis.append(n + i)
        elif signs[i] == ">=":
            slack[i] = -1
            artificial_var[i] = 1
            row += slack + artificial_var + [b[i]]
            basis.append(n + m + i)
            artificial.append(n + m + i)
        elif signs[i] == "=":
            artificial_var[i] = 1
            row += slack + artificial_var + [b[i]]
            basis.append(n + m + i)
            artificial.append(n + m + i)
        tableau.append(row)

    # Mục tiêu Phase I: Min tổng biến phụ
    num_vars = n + m + m
    obj_row = [0] * num_vars + [0]
    for a in artificial:
        obj_row[a] = 1
    tableau.append(obj_row)

    tableau = np.array(tableau, dtype=float)

    # Điều chỉnh hàng mục tiêu Phase I (do các biến phụ đang trong cơ sở)
    for idx, var in enumerate(basis):
        if var in artificial:
            tableau[-1, :] -= tableau[idx, :]

    steps = [tableau.copy()]

    # Phase I Simplex
    while True:
        if all(tableau[-1, :-1] >= -1e-8):
            break
        col = np.argmin(tableau[-1, :-1])
        if all(tableau[:-1, col] <= 1e-8):
            return None, None, steps, "Không giới nội"

        ratios = [tableau[i, -1] / tableau[i, col] if tableau[i, col] > 1e-8 else np.inf for i in range(m)]
        row = np.argmin(ratios)
        pivot = tableau[row, col]
        tableau[row, :] /= pivot
        for i in range(m + 1):
            if i != row:
                tableau[i, :] -= tableau[i, col] * tableau[row, :]

        basis[row] = col
        steps.append(tableau.copy())

    # Kiểm tra tính khả thi
    if abs(tableau[-1, -1]) > 1e-5:
        return None, None, steps, "Bài toán vô nghiệm (infeasible)"

    # Xóa biến phụ & bước vào Phase II
    tableau = tableau[:-1, :]
    tableau = np.delete(tableau, artificial, axis=1)
    basis = [var for var in basis if var not in artificial]

    # Thiết lập hàng mục tiêu gốc
    obj = [-ci for ci in c] + [0] * (tableau.shape[1] - len(c) - 1) + [0]
    tableau = np.vstack([tableau, obj])

    # Điều chỉnh mục tiêu theo biến cơ sở
    for i, var in enumerate(basis):
        tableau[-1, :] -= tableau[-1, var] * tableau[i, :]

    steps.append(tableau.copy())

    # Phase II Simplex
    while True:
        if all(tableau[-1, :-1] >= -1e-8):
            break
        col = np.argmin(tableau[-1, :-1])
        if all(tableau[:-1, col] <= 1e-8):
            return None, None, steps, "Không giới nội"

        ratios = [tableau[i, -1] / tableau[i, col] if tableau[i, col] > 1e-8 else np.inf for i in range(m)]
        row = np.argmin(ratios)
        pivot = tableau[row, col]
        tableau[row, :] /= pivot
        for i in range(m + 1):
            if i != row:
                tableau[i, :] -= tableau[i, col] * tableau[row, :]
        basis[row] = col
        steps.append(tableau.copy())

    # Lấy nghiệm tối ưu
    sol = np.zeros(n)
    for i, var in enumerate(basis):
        if var < n:
            sol[var] = tableau[i, -1]

    val = tableau[-1, -1]
    return sol, val, steps, "Tối ưu"


# ========== Vẽ miền chấp nhận 2D ========== #
def plot_feasible_region(A, b, c, sol=None, signs=None, variables=None):
    x = np.linspace(0, 10, 800)
    y = np.linspace(0, 10, 800)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    cond = np.ones_like(X, dtype=bool)

    if signs is None:
        signs = ["<="] * len(A)
    if variables is None or len(variables) < 2:
        st.warning("Không đủ biến để vẽ.")
        return

    for row, bound, sign in zip(A, b, signs):
        expr = row[0] * X + row[1] * Y
        if sign == "<=":
            cond &= expr <= bound
        elif sign == ">=":
            cond &= expr >= bound

        if row[1] != 0:
            y_line = (bound - row[0] * x) / row[1]
            ax.plot(x, y_line, label=f'{pretty_expr(row, variables)} {sign} {bound:g}')
        elif row[0] != 0:
            ax.axvline(bound / row[0], label=f'{pretty_expr(row, variables)} {sign} {bound:g}')

    ax.imshow(cond.astype(int), extent=(0, 10, 0, 10), origin='lower', cmap='Greys', alpha=0.3)

    if sol is not None and len(sol) >= 2:
        ax.plot(sol[0], sol[1], 'ro', label='Điểm tối ưu')
        ax.annotate(f'({sol[0]:.2f}, {sol[1]:.2f})', (sol[0], sol[1]), textcoords="offset points", xytext=(5, 5), color='red')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True)
    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[1])
    ax.legend(loc="upper right")
    st.pyplot(fig)

# ========== Vẽ 3D ========== #
def plot_3d_feasible_region(A, b, c, sol=None, variables=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, 20, 60)
    y = np.linspace(0, 20, 60)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    cond = np.ones_like(X, dtype=bool)

    for row, bound in zip(A, b):
        expr = row[0] * X + row[1] * Y
        cond &= expr <= bound
    Z[~cond] = np.nan
    Z[cond] = c[0] * X[cond] + c[1] * Y[cond]

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    ax.view_init(elev=30, azim=150)

    if sol is not None:
        opt_val = c[0] * sol[0] + c[1] * sol[1]
        ax.scatter(sol[0], sol[1], opt_val, color='red', s=50)
        ax.text(sol[0], sol[1], opt_val, "  Tối ưu", color='red')

    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[1])
    ax.set_zlabel('Z')
    st.pyplot(fig)

# ========== Helpers ========== #
def pretty_expr(coeffs, variables):
    terms = []
    for coef, var in zip(coeffs, variables):
        if coef == 0: continue
        elif coef == 1: terms.append(f"{var}")
        elif coef == -1: terms.append(f"-{var}")
        else: terms.append(f"{coef:g}{var}")
    return " + ".join(terms).replace("+-", "- ")

def parse_objective(raw_text):
    raw_text = raw_text.lower().replace(" ", "")
    mode = "max" if "max" in raw_text else "min"
    raw_text = re.sub(r'(max|min|minimize|maximize)', '', raw_text)
    raw_text = re.sub(r'^[+-]+', '', raw_text)  # Xoá dấu + hoặc - thừa ở đầu
    terms = re.findall(r'([+-]?\d*)?([a-zA-Z]\w*)', raw_text)
    variables, coeffs = [], []
    for coef_str, var in terms:
        coef = int(coef_str) if coef_str not in ["", "+", "-"] else int(coef_str + "1")
        variables.append(var)
        coeffs.append(coef)
    return mode, coeffs, variables


def parse_constraint(line, all_variables):
    line = line.replace("≤", "<=").replace("≥", ">=")
    parts = re.split(r'(<=|>=|=)', line.replace(" ", ""))
    if len(parts) != 3:
        return None
    expr, sign, rhs = parts
    terms = re.findall(r'([+-]?\d*)?([a-zA-Z]\w*)', expr)
    found_vars = set(var for _, var in terms)
    if not any(var in all_variables for var in found_vars):
        return None
    coeffs_dict = {v: 0 for v in all_variables}
    for coef_str, var in terms:
        coef = int(coef_str) if coef_str not in ["", "+", "-"] else int(coef_str + "1")
        if var in coeffs_dict:
            coeffs_dict[var] += coef
        else:
            coeffs_dict[var] = coef
    coeffs = [coeffs_dict[v] for v in all_variables]
    return coeffs, sign, float(rhs)

# ========== UI ========== #
st.set_page_config(page_title="LP Solver - Simplex", layout="centered")
st.title("📈 Simplex - Linear Programming Solver")

st.subheader("🎯 Hàm mục tiêu")
obj_str = st.text_input("Ví dụ: Max 3x + 2y", value="Max 3x + 2y")

st.subheader("📏 Ràng buộc (cách nhau bằng dấu phẩy ',')")
constraints_input = st.text_input("Ví dụ: 2x + 3y ≤ 10, x + y <= 6", value="2x + 3y ≤ 10, x + y ≤ 6")
constraint_lines = re.split(r'\s*,\s*', constraints_input.strip())

st.subheader("🧩 Ràng buộc dấu biến (cách nhau bằng dấu phẩy ',')")
signs_input = st.text_input("Ví dụ: x ≥ 0, y >=0", value="x ≥ 0, y ≥ 0")
sign_lines = re.split(r'\s*,\s*', signs_input.strip())

st.subheader("📋 Xác nhận dữ liệu đầu vào")
try:
    mode, c, obj_vars = parse_objective(obj_str)

    all_vars = obj_vars.copy()
    for line in constraint_lines:
        terms = re.findall(r'[a-zA-Z]\w*', line)
        for var in terms:
            if var not in all_vars:
                all_vars.append(var)

    if len(all_vars) > len(c):
        c += [0] * (len(all_vars) - len(c))

    st.markdown(f"**Loại bài toán:** `{mode.upper()}`")
    st.markdown(f"**Hàm mục tiêu:** `{mode} Z = {pretty_expr(c, all_vars)}`")

    st.markdown("**Danh sách ràng buộc:**")
    valid_constraints = []
    signs = []
    for line in constraint_lines:
        if line.strip():
            parsed = parse_constraint(line, all_vars)
            if parsed:
                coeffs, sign, rhs = parsed
                expr = pretty_expr(coeffs, all_vars)
                st.write(f"{expr} {sign} {rhs:g}")
                valid_constraints.append((coeffs, rhs))
                signs.append(sign)
            else:
                st.error(f"Ràng buộc không hợp lệ hoặc không có biến trùng khớp: {line}")

    st.markdown("**Ràng buộc dấu biến:**")
    for i in range(len(all_vars)):
        if i < len(sign_lines):
            raw = sign_lines[i].strip().replace("≤", "<=").replace("≥", ">=").lower()
            if ">=" in raw:
                st.write(f"{all_vars[i]} ≥ 0")
            elif "<=" in raw:
                st.write(f"{all_vars[i]} ≤ 0")
            elif "tựdo" in raw or "free" in raw:
                st.write(f"{all_vars[i]} tự do")
            else:
                st.write(f"{all_vars[i]} tự do")
        else:
            st.write(f"{all_vars[i]} tự do")
except Exception as e:
    st.error(f"Lỗi khi phân tích dữ liệu: {e}")

if st.button("🚀 Giải bài toán"):
    try:
        mode, c, obj_vars = parse_objective(obj_str)

        all_vars = obj_vars.copy()
        for line in constraint_lines:
            terms = re.findall(r'[a-zA-Z]\w*', line)
            for var in terms:
                if var not in all_vars:
                    all_vars.append(var)

        if len(all_vars) > len(c):
            c += [0] * (len(all_vars) - len(c))

        A, b = [], []
        for line in constraint_lines:
            if line.strip() == "":
                continue
            parsed = parse_constraint(line, all_vars)
            if not parsed:
                st.error(f"Ràng buộc không hợp lệ: {line}")
                st.stop()
            coeffs, sign, rhs = parsed
            A.append(coeffs)
            b.append(rhs)

        is_min = mode == "min"
        if is_min:
            c = [-x for x in c]

        sol, val, steps, status = simplex_full_phase(c, A, b, signs)

        st.success(f"🔎 Trạng thái: {status}")
        if sol is not None:
            st.subheader("✅ Nghiệm tối ưu")
            for i in range(len(sol)):
                st.write(f"{all_vars[i]} = {sol[i]:.2f}")
            if is_min:
                val = -val
            st.write(f"🎯 Giá trị tối ưu: {val:.2f}")

            st.subheader("📄 Các bước lặp:")
            for i, step in enumerate(steps):
                st.markdown(f"**Bước {i+1}**")
                st.dataframe(np.round(step, 2))

            if len(c) >= 2:
                st.subheader("📊 Đồ thị nghiệm (2D)")
                plot_feasible_region(A, b, c, sol, signs, all_vars)

                st.subheader("🧱 Đồ thị 3D (Z theo 2 biến đầu)")
                plot_3d_feasible_region(A, b, c, sol, all_vars)
        else:
            st.error("Không tìm được nghiệm.")
    except Exception as e:
        st.error(f"Lỗi: {e}")
