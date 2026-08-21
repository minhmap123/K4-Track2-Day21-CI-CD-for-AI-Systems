# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Nguyễn Hoàng Bảo Minh |
| MSSV | 2A202601626  |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/minhmap123/K4-Track2-Day21-CI-CD-for-AI-Systems |
| Ngày nộp | 2026-08-21 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7149 | 0.8740 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`.

**Lý do:** Bộ tham số này mang lại `f1_score` cao (0.7149), đủ cấu thành điều kiện để vượt qua ngưỡng (0.65). Mặc dù bộ siêu tham số thư 3 cũng cho điểm bằng nhưng lại phức tạp hơn (nhiều estimators và depth sâu hơn) dẫn đến chi phí tính toán cao và nguy cơ overfitting. Ta cũng nhìn thấy độ nhạy của learning_rate qua lần 2 khi giảm cả hai tham số khiến mô hình không hội tụ đủ tốt (F1 chỉ đạt 0.6051).

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Trong tập dữ liệu phân loại thu nhập này, tính mất cân bằng là rất rõ ràng (chỉ 24,8% mẫu nằm ở lớp thu nhập cao tức >50K). Hệ quả của sự mất cân bằng này là bất kì mô hình bù nhìn nào đoán tất cả các mẫu là "thu nhập thấp" vẫn sẽ giành được `accuracy` lên tới ~0.752. Nếu dùng accuracy để đánh giá, chúng ta dễ bị ảo tưởng vào sức mạnh của một mô hình hoàn toàn vô dụng.
Bằng cách đánh giá qua `f1_score` (chuyên đo đạc trên lớp dương), ta đo lường sát sao hơn khả năng bắt và dự đoán đúng lớp >50K của mô hình. Trong `train.py`, ta không truyền các tham số như `average="weighted"` vào `f1_score` để tránh hiện tượng nhãn đa số kéo f1 lên, đảm bảo mô hình phản ánh chuẩn xác chất lượng.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lỗi phiên bản scikit-learn sinh ra AttributeError khi khởi động API qua uvicorn trên VM. | Phiên bản gốc khi train model ở local là (1.4.2) cài qua UV, nhưng `pip3 install` trên VM mặc định tải bản mới nhất (1.7.x) xung đột cách load `.joblib`. | Ssh trực tiếp vào GCE VM và chạy cài đè phiên bản đích danh `pip3 install scikit-learn==1.4.2`. |
| Pipeline Github Action CI/CD bị kẹt ở bước "Pull data with DVC". | Tài khoản Google Cloud bị dính chính sách `iam.disableServiceAccountKeyCreation` của Organization, chặn tải mã JSON khiến biến bảo mật bị thiếu. | Truy cập GCP Management Console điều chỉnh ghi đè (Override) Policy, chuyển Enforcement thành Off để sinh khóa thủ công vào Github Secrets. |
| Server FastAPI không khởi động do systemctl báo lỗi CHDIR thất bại. | Biến `$USER` truyền vào file setup `income-api.service` khi chạy bằng bash sudo đã bị thay thế thành `root`, làm hỏng đường dẫn thư mục làm việc `/home/minu2k5`. | Đính kèm rõ ràng tên User gốc (`minu2k5`) vào file cấu hình service thay vì gán động. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7014 | 0.8740 |

**Nhận xét:** f1 giảm nhẹ ~0.01 do dữ liệu thêm vào có cùng điểm phân phối với dữ liệu cũ, không mang thêm thông tin mới đột phá nào. Điều quan trọng nhất là toàn bộ luồng huấn luyện liên tục (Continuous Training) CI/CD được thực hiện tự động hoàn toàn không cần can thiệp thủ công.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

Mình đã hoàn thành tự động toàn bộ 5/5 thử thách nâng cao:

**Bonus 1: Tracking MLflow Từ Xa Với DagsHub**
Thay vì lưu cục bộ, toàn bộ luồng huấn luyện CI/CD Github Actions đã được nạp Credentials để tự động bắn thẳng Log/Metrics từ máy chủ của Google Cloud lên bảng theo dõi từ xa của nền tảng DagsHub (Sử dụng remote server URI).

**Bonus 2: Điều Chỉnh Ngưỡng Quyết Định (Thresholding)**
Thay vì ngưỡng 0.5, thuật toán quét ngưỡng (từ 0.1 tới 0.9) đã tìm được `best_threshold` để tối ưu `f1_score`. Thực tế cho thấy, F1 ở ngưỡng mặc định 0.5 bị hụt giảm đáng kể khi dữ liệu bất đối xứng; trong khi ở ngưỡng tối ưu, F1 được đẩy lên giới hạn tốt nhất có thể, giúp bảo vệ tính toàn vẹn của model khi hoạt động thực tế.

**Bonus 3: Báo Cáo Precision / Recall Tự Động**
Luồng CI/CD đã tự động sinh `detail.txt` chứa Precision, Recall và Confusion Matrix cho từng phiên bản. 
- *Phân tích Trade-off:* Trong bài toán tìm kiếm tập khách hàng có thu nhập cao (>50K) này, nếu ta **gán nhầm (Precision thấp)**, doanh nghiệp sẽ lãng phí chi phí marketing cho nhóm khách nghèo. Nếu ta **bỏ sót (Recall thấp)**, hệ thống sẽ bỏ qua người thực sự giàu và làm hụt doanh thu tiềm năng khổng lồ. Việc ưu tiên Precision hay Recall sẽ phụ thuộc hoàn toàn vào ngân sách chiến dịch: Nếu ngân sách hẹp, hãy tối ưu Precision.

**Bonus 4: Hoàn Trả Về Phiên Bản Trước (Safe Rollback Quality Gate)**
Đã bổ sung trong luồng `cicd.yml` cơ chế Load tự động `old_report.json` và so sánh. Khi và chỉ khi F1 của Model vừa Train lớn hơn hoặc bằng F1 của Model cũ đang chạy sản xuất thì nút Deploy (Release) mới được bật. Nếu F1 mới < F1 cũ, Quality Gate sẽ bị rớt đỏ.

**Bonus 5: Cảnh Báo Lệch Lạc Dữ Liệu (Data Drift Check)**
Đã nhúng mã kiểm tra độ lệch tỉ lệ phân bổ Label trong Python trước khi Train. Nếu tỉ lệ nhãn dương >50K bất ngờ bật chênh quá >5% so với phân phối dự kiến gốc (24.8%), nó sẽ được cảnh báo đỏ trên Terminal log để Data Engineer can thiệp làm sạch dữ liệu.
