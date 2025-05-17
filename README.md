Bài toán Quy hoạch tuyến tính bằng thuật toán Simplex tự viết, chạy trên Streamlit

simplex_app/
│
├── app.py              ← file chính Streamlit
├── simplex_solver.py   ← nơi cài đặt thuật toán Simplex
├── plot_lp.py          ← vẽ hình nếu n = 2
└── requirements.txt    ← các thư viện cần cài

> pip install streamlit sympy matplotlib plotly numpy
Viết thuật toán Simplex cơ bản (file simplex_solver.py)
Vẽ nghiệm nếu n = 2 (file plot_lp.py)
> python -m streamlit run app.py

Cải tiến parser để hỗ trợ các biểu thức như max-3x1+2x2 hoặc max -3x1 + 2x2
Hỗ trợ các ký hiệu toán học như ≤ và ≥.
Xử lý bài toán không giới nội / vô nghiệm bằng cách kiểm tra trạng thái và các điều kiện khi không có biến vào được chọn.