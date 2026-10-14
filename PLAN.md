# Kế hoạch dự án Telco Customer Churn

Mục tiêu: học quy trình phân tích dữ liệu và xây dựng mô hình phân loại khách hàng đã rời dịch vụ (`Churn`).

1. Tải CSV công khai từ Kaggle; nếu không truy cập được, dùng bản mẫu do IBM phát hành và ghi rõ nguồn thực tế, SHA-256.
2. Kiểm tra schema, nhãn, mã khách hàng trùng, dữ liệu thiếu; chuyển `TotalCharges` sang số và giữ giá trị thiếu cho pipeline xử lý.
3. Phân tích tỷ lệ churn theo hợp đồng, thời gian sử dụng, phí hàng tháng. Những liên hệ này không chứng minh quan hệ nhân quả.
4. Chia 60% train, 20% validation, 20% test với stratification, seed 42. Fit imputer, scaler và encoder chỉ trên train.
5. Huấn luyện baseline luôn dự đoán không churn, Logistic Regression và Random Forest. Chọn ngưỡng cho mỗi mô hình theo F1 trên validation; chọn mô hình bằng validation F1. Test chỉ dùng để báo cáo sau khi khóa lựa chọn.
6. Lưu notebook đã chạy, biểu đồ PNG, metrics JSON/CSV, báo cáo Markdown, pipeline joblib, danh sách phân chia và lệnh dự đoán CSV.

Tiêu chí hoàn thành: xử lý toàn bộ dữ liệu hợp lệ, các tập không trùng khách hàng, có kết quả test thực tế, notebook không lỗi và pipeline đã lưu cho kết quả dự đoán nhất quán.

Giới hạn: dữ liệu mẫu dùng để học; nhãn mô tả khách hàng đã rời đi trong tháng gần nhất. Không có mốc thời gian để kiểm chứng mô hình dự báo churn tương lai hoặc hiệu quả triển khai thực tế.

## Trạng thái thực hiện

Đã hoàn thành cả 6 bước và chạy toàn bộ notebook không lỗi. Dữ liệu tải trực tiếp từ Kaggle: 7.043 khách hàng, 21 cột. Các tập thực tế: train 4.225, validation 1.409, test 1.409.

Logistic Regression được chọn bằng F1 validation 0,630 tại ngưỡng 0,37. Test: accuracy 0,769; precision 0,552; recall 0,690; F1 0,614; ROC-AUC 0,843. Random Forest được đánh giá cùng quy trình; số liệu đầy đủ trong `reports/test_metrics.csv`.

10 kiểm tra tính toàn vẹn đã đạt; xem `reports/verification.json`. Notebook có kết quả đã chạy, bốn biểu đồ, báo cáo, pipeline và ví dụ dự đoán CSV đã được lưu.
