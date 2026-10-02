#!/usr/bin/env python3
"""
benchmark.py - Script huấn luyện & benchmark mô hình LightGBM cho Lab 16 (Bước 4.4)
Dataset: Credit Card Fraud Detection (mlg-ulb/creditcardfraud)

Nhiệm vụ theo yêu cầu Bước 4.4:
1. Load dataset và tách tập train/test (stratify).
2. Huấn luyện LGBMClassifier để phát hiện giao dịch gian lận.
3. Đo thời gian load data và thời gian training.
4. Đánh giá model trên tập test: AUC-ROC, Accuracy, F1-Score, Precision, Recall.
5. Đo inference latency (dự đoán 1 dòng) và throughput (dự đoán 1000 dòng).
6. Ghi toàn bộ kết quả ra file benchmark_result.json.
7. In bảng kết quả chuẩn định dạng Markdown để điền vào báo cáo.
"""

import os
import sys
import time
import json
import argparse
import platform
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
import lightgbm as lgb


def find_dataset(custom_path=None):
    """Tìm đường dẫn file creditcard.csv"""
    candidates = []
    if custom_path:
        candidates.append(os.path.abspath(os.path.expanduser(custom_path)))
    
    # Các vị trí phổ biến theo hướng dẫn Lab
    candidates.extend([
        os.path.expanduser("~/ml-benchmark/creditcard.csv"),
        "/home/ubuntu/ml-benchmark/creditcard.csv",
        "creditcard.csv",
        os.path.join(os.getcwd(), "creditcard.csv"),
        os.path.join(os.getcwd(), "ml-benchmark", "creditcard.csv"),
        os.path.expanduser("~/creditcard.csv")
    ])

    for path in candidates:
        if os.path.isfile(path):
            return path
            
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark LightGBM trên CPU Node cho Credit Card Fraud Detection (Lab 16)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Đường dẫn tới file creditcard.csv (mặc định tự động tìm)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_result.json",
        help="Đường dẫn file JSON lưu kết quả (mặc định: benchmark_result.json)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Tỷ lệ tập test (mặc định: 0.2)"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Số lượng cây n_estimators (mặc định: 100)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (mặc định: 42)"
    )
    parser.add_argument(
        "--latency-runs",
        type=int,
        default=500,
        help="Số lần lặp đo single-row latency (mặc định: 500)"
    )
    parser.add_argument(
        "--throughput-runs",
        type=int,
        default=50,
        help="Số lần lặp đo batch 1000 throughput (mặc định: 50)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🚀 LAB 16: BENCHMARK LIGHTGBM TRÊN COMPUTE NODE (CPU)")
    print("=" * 70)
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {platform.python_version()} | LightGBM: {lgb.__version__}")
    print(f"CPU Cores: {os.cpu_count()}")
    print("-" * 70)

    # 1. Tìm dataset
    data_path = find_dataset(args.data_path)
    if not data_path:
        print("❌ LỖI: Không tìm thấy file 'creditcard.csv'!")
        print("Vui lòng tải dataset theo Bước 4.3 trong README:")
        print("  kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/")
        print("Hoặc truyền cờ --data-path: python3 benchmark.py --data-path /đường/dẫn/creditcard.csv")
        sys.exit(1)

    print(f"📂 Dataset file: {data_path}")

    # ==========================================
    # BƯỚC 1: LOAD DỮ LIỆU & ĐO THỜI GIAN
    # ==========================================
    print("\n[1/5] Đang đọc dữ liệu từ CSV...")
    t_load_start = time.perf_counter()
    df = pd.read_csv(data_path)
    load_time_sec = time.perf_counter() - t_load_start
    
    total_samples = len(df)
    total_features = df.shape[1] - 1  # trừ cột Class
    fraud_samples = int(df['Class'].sum())
    normal_samples = total_samples - fraud_samples
    fraud_ratio = (fraud_samples / total_samples) * 100

    print(f"  ✓ Thời gian load data: {load_time_sec:.4f} giây")
    print(f"  ✓ Tổng số dòng: {total_samples:,} dòng | Số đặc trưng: {total_features}")
    print(f"  ✓ Giao dịch bình thường (Class 0): {normal_samples:,} ({100 - fraud_ratio:.2f}%)")
    print(f"  ✓ Giao dịch gian lận (Class 1): {fraud_samples:,} ({fraud_ratio:.4f}%)")

    # ==========================================
    # BƯỚC 2: TÁCH TẬP TRAIN / TEST
    # ==========================================
    print("\n[2/5] Đang chia tập Train / Test (80% / 20%, stratify=y)...")
    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )
    print(f"  ✓ Kích thước Train: {len(X_train):,} dòng | Test: {len(X_test):,} dòng")

    # ==========================================
    # BƯỚC 3: HUẤN LUYỆN MÔ HÌNH LIGHTGBM
    # ==========================================
    print("\n[3/5] Đang khởi tạo và huấn luyện LGBMClassifier...")
    clf = lgb.LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        random_state=args.random_state,
        n_jobs=-1,
        verbosity=-1,
        objective='binary',
        metric='auc'
    )

    t_train_start = time.perf_counter()
    try:
        callbacks = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=0)
        ]
        clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc',
            callbacks=callbacks
        )
    except Exception:
        # Fallback nếu phiên bản LightGBM không nhận callbacks cú pháp này
        clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc'
        )

    training_time_sec = time.perf_counter() - t_train_start

    # Lấy best iteration
    best_iteration = getattr(clf, 'best_iteration_', None)
    if best_iteration is None or best_iteration <= 0:
        best_iteration = clf.n_estimators
    else:
        best_iteration = int(best_iteration)

    print(f"  ✓ Thời gian training: {training_time_sec:.4f} giây")
    print(f"  ✓ Best iteration: {best_iteration}")

    # ==========================================
    # BƯỚC 4: ĐÁNH GIÁ TRÊN TẬP TEST
    # ==========================================
    print("\n[4/5] Đang đánh giá model trên tập Test...")
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc_roc = float(roc_auc_score(y_test, y_pred_proba))
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))

    print(f"  ✓ AUC-ROC:   {auc_roc:.4f}")
    print(f"  ✓ Accuracy:  {accuracy:.6f} ({accuracy * 100:.3f}%)")
    print(f"  ✓ F1-Score:  {f1:.4f}")
    print(f"  ✓ Precision: {precision:.4f}")
    print(f"  ✓ Recall:    {recall:.4f}")

    # ==========================================
    # BƯỚC 5: ĐO INFERENCE LATENCY & THROUGHPUT
    # ==========================================
    print("\n[5/5] Đang đo inference latency (1 dòng) & throughput (1000 dòng)...")
    
    # 5.1 Đo Latency (1 dòng)
    single_row = X_test.iloc[0:1]
    # Warmup
    for _ in range(10):
        _ = clf.predict_proba(single_row)

    latencies = []
    for _ in range(args.latency_runs):
        t0 = time.perf_counter()
        _ = clf.predict_proba(single_row)
        latencies.append(time.perf_counter() - t0)

    avg_latency_s = float(np.mean(latencies))
    avg_latency_ms = avg_latency_s * 1000.0

    # 5.2 Đo Throughput (1000 dòng)
    batch_size = 1000
    if len(X_test) >= batch_size:
        batch_1000 = X_test.iloc[:batch_size]
    else:
        batch_1000 = X_test

    # Warmup
    for _ in range(5):
        _ = clf.predict_proba(batch_1000)

    batch_times = []
    for _ in range(args.throughput_runs):
        t0 = time.perf_counter()
        _ = clf.predict_proba(batch_1000)
        batch_times.append(time.perf_counter() - t0)

    avg_batch_time_s = float(np.mean(batch_times))
    avg_batch_time_ms = avg_batch_time_s * 1000.0
    throughput_rows_per_sec = float(batch_size / avg_batch_time_s) if avg_batch_time_s > 0 else 0.0

    print(f"  ✓ Latency (1 row): {avg_latency_ms:.3f} ms (trung bình {args.latency_runs} lần chạy)")
    print(f"  ✓ Thời gian chạy 1000 rows: {avg_batch_time_ms:.2f} ms")
    print(f"  ✓ Throughput: {throughput_rows_per_sec:,.1f} rows/giây")

    # ==========================================
    # GHI KẾT QUẢ RA FILE JSON
    # ==========================================
    result_data = {
        "load_time_seconds": round(load_time_sec, 4),
        "training_time_seconds": round(training_time_sec, 4),
        "best_iteration": best_iteration,
        "auc_roc": round(auc_roc, 4),
        "accuracy": round(accuracy, 6),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "inference_latency_1_row_ms": round(avg_latency_ms, 3),
        "inference_batch_1000_rows_ms": round(avg_batch_time_ms, 2),
        "inference_throughput_rows_per_sec": round(throughput_rows_per_sec, 1),
        "table_summary": {
            "Thời gian load data": f"{load_time_sec:.4f} s",
            "Thời gian training": f"{training_time_sec:.4f} s",
            "Best iteration": best_iteration,
            "AUC-ROC": f"{auc_roc:.4f}",
            "Accuracy": f"{accuracy:.6f}",
            "F1-Score": f"{f1:.4f}",
            "Precision": f"{precision:.4f}",
            "Recall": f"{recall:.4f}",
            "Inference latency (1 row)": f"{avg_latency_ms:.3f} ms",
            "Inference throughput (1000 rows)": f"{avg_batch_time_ms:.2f} ms ({throughput_rows_per_sec:,.0f} rows/s)"
        },
        "dataset_info": {
            "dataset_file": data_path,
            "total_samples": total_samples,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": total_features,
            "fraud_samples": fraud_samples,
            "fraud_ratio_percent": round(fraud_ratio, 4)
        },
        "system_info": {
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "cpu_cores": os.cpu_count(),
            "python_version": platform.python_version(),
            "lightgbm_version": lgb.__version__
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    output_path = os.path.abspath(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Đã lưu kết quả chi tiết vào: {output_path}")

    # ==========================================
    # IN BẢNG KẾT QUẢ MARKDOWN CHO BÁO CÁO
    # ==========================================
    print("\n" + "=" * 70)
    print("📋 BẢNG KẾT QUẢ BENCHMARK (Sao chép vào báo cáo Lab 16):")
    print("=" * 70)
    print("| Metric | Kết quả |")
    print("|---|---|")
    print(f"| Thời gian load data | {load_time_sec:.4f} s |")
    print(f"| Thời gian training | {training_time_sec:.4f} s |")
    print(f"| Best iteration | {best_iteration} |")
    print(f"| AUC-ROC | {auc_roc:.4f} |")
    print(f"| Accuracy | {accuracy:.6f} |")
    print(f"| F1-Score | {f1:.4f} |")
    print(f"| Precision | {precision:.4f} |")
    print(f"| Recall | {recall:.4f} |")
    print(f"| Inference latency (1 row) | {avg_latency_ms:.3f} ms |")
    print(f"| Inference throughput (1000 rows) | {avg_batch_time_ms:.2f} ms ({throughput_rows_per_sec:,.0f} rows/s) |")
    print("=" * 70)
    print("✅ Hoàn thành Bước 4.4 thành công!")


if __name__ == "__main__":
    main()
