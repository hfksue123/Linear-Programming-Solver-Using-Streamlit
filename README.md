# 🧮 Linear Programming Solver with Simplex Algorithm (Streamlit App)

An application for solving **Linear Programming (LP)** problems using the **Simplex algorithm**, built with **Streamlit**.

## 💯 Features

- Input objective function in natural form: `Max 3x + 2y` or `Min 5x + 4y`
- Easily input constraints: `2x + 3y ≤ 10, x + y ≤ 6`
- Supports variable bounds: `x ≥ 0, y ≥ 0`
- Automatically detects:
  - Infeasible problems
  - Unbounded problems
  - Optimal solutions (if exist)
- Displays all steps of the Simplex method (Phase I and Phase II)
- Plots:
  - 2D feasible region
  - 3D objective plane visualization
- User-friendly web interface powered by **Streamlit**

## 🛠️
### Install library
```bash
pip install streamlit matplotlib numpy scipy
```
### Run local
```bash
streamlit run app.py
```
## 🚀
### Deploy on Git
```bash
git init
git remote add origin https://github.com/yourusername/your-repo.git
git add .
git commit -m "Initial commit"
git push -u origin master
```
### Deploy on Streamlit
Make sure to have a file named `requirements.txt` in your project directory with nessary dependencies.

Visit https://streamlit.io/cloud and click on Free Deploy to deploy your app with Github.

## ⚠️
With Free Version, the app will be slept after 1 hour if you don't use it. But don't worry, you can wake him up anytime by clicking the button and it could take up 30 secs.




