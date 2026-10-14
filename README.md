# Telco Customer Churn

Dự án nhập môn: phân tích khách hàng rời dịch vụ, so sánh Logistic Regression / Random Forest, lưu pipeline và dự đoán từ CSV.

## Bắt đầu

Mở `telco_churn.ipynb` để xem các bước và kết quả đã chạy. Đọc `reports/REPORT.md` để xem báo cáo ngắn, `PLAN.md` để xem kế hoạch và `reports/test_metrics.csv` để xem điểm test.

Trong PowerShell:

```powershell
cd 'D:\Downloads\AI Engineer\AI1\telco-churn'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe telco_churn.py
```

Môi trường hiện tại đã có các thư viện nên có thể chạy thẳng `python telco_churn.py`. Requirements ghi lại phiên bản môi trường thực hiện. Notebook cần chọn kernel dùng cùng môi trường Python. Sinh lại notebook trống bằng `python make_notebook.py`; lệnh này ghi đè notebook, nên chỉ dùng nếu bạn chủ động muốn bỏ các output cũ.

Chạy toàn bộ notebook và lưu output bằng `python execute_notebook.py`. Kiểm chứng checksum, phân chia dữ liệu, preprocessing chỉ học từ train, dự đoán từ mô hình đã lưu và notebook bằng `python verify_project.py`; kết quả ghi trong `reports/verification.json`.

## Dự đoán

```powershell
python predict.py --input data/example_customers.csv --output reports/new_predictions.csv
```

Ví dụ có ba khách hàng từ tập test, chỉ để minh họa. Thay input bằng CSV của bạn có đủ các cột trong `models/model_metadata.json`. `predicted_churn`: 1 là rời dịch vụ, 0 là ở lại. Ngưỡng cố định lấy từ validation, không mặc định 0,50. File dự đoán không cần có `Churn`; có thể có `customerID`.

## Tệp dự án

- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`: dữ liệu gốc.
- `data/source.json`: nguồn tải thực tế và SHA-256.
- `telco_churn.py`: kiểm tra, làm sạch, chia tập, EDA, huấn luyện và báo cáo.
- `telco_churn.ipynb`: notebook học từng bước, có output thực tế.
- `predict.py`: dự đoán bằng pipeline đã lưu.
- `models/`: pipeline cùng ngưỡng, schema, phiên bản thư viện.
- `reports/`: metrics, lựa chọn ngưỡng, test predictions, splits, báo cáo và bốn biểu đồ PNG.

## Phương pháp và giới hạn

Chia stratified 60% train / 20% validation / 20% test, seed 42. Preprocessing chỉ học từ train. Mô hình và ngưỡng được chọn bằng validation F1 rồi giữ nguyên khi đánh giá test. Baseline luôn dự đoán không churn giúp nhận diện accuracy gây hiểu nhầm với nhãn mất cân bằng.

Nguồn: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Nếu Kaggle không truy cập được, downloader dùng [kho dữ liệu mẫu IBM](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) và ghi nguồn thực tế. IBM mô tả [Telco Customer Churn là dữ liệu mẫu của công ty giả định](https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn). Không xem kết quả trên dữ liệu mẫu là bằng chứng hiệu quả kinh doanh. Nhãn mô tả churn tháng vừa qua; dữ liệu không có mốc thời gian đủ để chứng minh dự báo churn tương lai. Xác suất chưa được hiệu chỉnh.
