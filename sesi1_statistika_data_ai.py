from datetime import datetime, timedelta
import math
import random
import statistics

def generate_company_dataset(n=350):
    random.seed(2026)
    base_date = datetime(2025, 1, 1)
    data = []

    for i in range(1, n + 1):
        inv_date = base_date + timedelta(days=i)
        ret_date = (
            inv_date - timedelta(days=random.randint(1, 5))
            if random.random() < 0.05
            else None
        )

        invoice_amt = (
            None
            if random.random() < 0.04
            else random.uniform(5_000_000, 80_000_000)
        )

        if random.random() < 0.12:
            discount_pct = random.uniform(0.35, 0.60)
        else:
            discount_pct = random.uniform(0.0, 0.15)

        hist_late = random.choices([0, 1, 2, 3, 4], weights=[45, 25, 15, 10, 5])[0]
        credit_limit = random.choice([30_000_000, 60_000_000, 100_000_000])

        eff_amt = invoice_amt if invoice_amt is not None else 25_000_000
        net_profit = eff_amt * (0.28 - (discount_pct * 0.65))

        logit = (
            -2.4
            + 1.9 * (eff_amt / credit_limit)
            + 0.85 * hist_late
            + 2.8 * discount_pct
        )
        prob_late = 1.0 / (1.0 + math.exp(-logit))
        is_late = 1 if random.random() < prob_late else 0

        if is_late == 1:
            days_to_pay = random.randint(31, 90)
            event_unpaid_at_30d = 1 
        else:
            days_to_pay = random.randint(1, 30)
            event_unpaid_at_30d = 0

        data.append({
            "invoice_id": f"INV-{i:04d}",
            "invoice_date": inv_date,
            "return_date": ret_date,
            "invoice_amount": invoice_amt,
            "discount_rate": discount_pct,
            "net_profit": net_profit,
            "hist_late_count": hist_late,
            "credit_limit": credit_limit,
            "days_after_due": days_to_pay,
            "is_late_payment": is_late,
            "actual_future_collection_fee": (
                days_to_pay * 50000 if is_late else 0
            ),  
        })

    data.extend(data[:8])
    return data


def prepare_and_clean_data(raw_data):
    print("=" * 85)
    print("TAHAP 1: DATA PREPARATION & PEMBERSIHAN DATA TRAPS (CRISP-DM)")
    print("=" * 85)
    print(f"Total observasi awal (sebelum pembersihan) : {len(raw_data)} baris")

    unique_dict = {row["invoice_id"]: dict(row) for row in raw_data}
    dedup_data = list(unique_dict.values())

    valid_amts = [
        r["invoice_amount"]
        for r in dedup_data
        if r["invoice_amount"] is not None
    ]
    median_amt = statistics.median(valid_amts)

    clean_data = []
    impossible_orders = 0

    for r in dedup_data:
        if r["return_date"] is not None and r["return_date"] < r["invoice_date"]:
            impossible_orders += 1
            continue
        if r["invoice_amount"] is None:
            r["invoice_amount"] = median_amt
            r["net_profit"] = median_amt * (0.28 - (r["discount_rate"] * 0.65))

        r.pop("actual_future_collection_fee", None)
        clean_data.append(r)

    print(f"- Duplikasi dihapus                        : {len(raw_data) - len(dedup_data)} baris")
    print(f"- Impossible Order (Retur < Beli) dihapus  : {impossible_orders} baris")
    print(f"- Imputasi Missing Value (Median)          : Rp {median_amt:,.2f}")
    print(f"Total observasi bersih siap analisis       : {len(clean_data)} baris\n")
    return clean_data

def multivariate_anomaly_and_hypothesis_test(data):
    print("=" * 85)
    print("TAHAP 2: DETEKSI ANOMALI (MAHALANOBIS DISTANCE) & UJI HIPOTESIS WELCH'S T-TEST")
    print("=" * 85)

    x1 = [r["invoice_amount"] / 1e6 for r in data]
    x2 = [r["discount_rate"] * 100.0 for r in data]
    n = len(data)

    mean_x1, mean_x2 = statistics.mean(x1), statistics.mean(x2)
    var_x1, var_x2 = statistics.variance(x1), statistics.variance(x2)
    cov_x1x2 = sum(
        (x1[i] - mean_x1) * (x2[i] - mean_x2) for i in range(n)
    ) / (n - 1)

    det_cov = (var_x1 * var_x2) - (cov_x1x2**2)
    inv_00 = var_x2 / det_cov
    inv_01 = -cov_x1x2 / det_cov
    inv_11 = var_x1 / det_cov

    chi2_critical_df2 = 9.210
    anomalies = []
    normals = []

    for i, r in enumerate(data):
        d1 = x1[i] - mean_x1
        d2 = x2[i] - mean_x2
        d_squared = (d1**2 * inv_00) + (2 * d1 * d2 * inv_01) + (d2**2 * inv_11)
        r["mahalanobis_d2"] = d_squared
        if d_squared > chi2_critical_df2 or r["discount_rate"] > 0.30:
            r["is_anomaly"] = 1
            anomalies.append(r)
        else:
            r["is_anomaly"] = 0
            normals.append(r)

    print(f"Matriks Kovarians [Tagihan(Juta), Diskon(%)]:")
    print(f"  Var(X1) = {var_x1:.2f} | Var(X2) = {var_x2:.2f} | Cov(X1,X2) = {cov_x1x2:.2f}")
    print(f"Jumlah Transaksi Normal             : {len(normals)} transaksi")
    print(f"Jumlah Transaksi Anomali (Outlier)  : {len(anomalies)} transaksi (D^2 > {chi2_critical_df2} atau Diskon > 30%)\n")

    prof_norm = [r["net_profit"] / 1e6 for r in normals]
    prof_anom = [r["net_profit"] / 1e6 for r in anomalies]

    m1, m2 = statistics.mean(prof_norm), statistics.mean(prof_anom)
    v1, v2 = statistics.variance(prof_norm), statistics.variance(prof_anom)
    n1, n2 = len(prof_norm), len(prof_anom)

    se_diff = math.sqrt((v1 / n1) + (v2 / n2))
    t_stat = (m1 - m2) / se_diff

    df_welch = ((v1 / n1 + v2 / n2) ** 2) / (
        ((v1 / n1) ** 2 / (n1 - 1)) + ((v2 / n2) ** 2 / (n2 - 1))
    )
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))

    print("--- Hasil Uji Hipotesis Komparatif Laba Bersih (Welch's t-test) ---")
    print("H0: Rata-rata Net Profit transaksi Normal == transaksi Anomali Diskon")
    print("H1: Rata-rata Net Profit transaksi Normal != transaksi Anomali Diskon")
    print(f"Rata-rata Profit Normal  : Rp {m1:.2f} Juta (SD = {math.sqrt(v1):.2f})")
    print(f"Rata-rata Profit Anomali : Rp {m2:.2f} Juta (SD = {math.sqrt(v2):.2f})")
    print(f"Statistik Uji t          : {t_stat:.4f} (df = {df_welch:.1f})")
    print(f"p-value                  : {p_value:.6e}")
    print(
        f"Kesimpulan Statistik     : {'Tolak H0 (Berbeda Signifikan pada alpha=0.05)' if p_value < 0.05 else 'Gagal Tolak H0'}\n"
    )

def logistic_regression_and_survival_analysis(data):
    print("=" * 85)
    print("TAHAP 3: PEMODELAN REGRESI LOGISTIK BINER & ANALISIS SURVIVAL (KAPLAN-MEIER)")
    print("=" * 85)

    X = []
    y = []
    for r in data:
        X.append([
            1.0,  
            r["invoice_amount"] / r["credit_limit"],
            float(r["hist_late_count"]),
            float(r["is_anomaly"]),
        ])
        y.append(r["is_late_payment"])

    weights = [0.0, 0.0, 0.0, 0.0]
    lr = 0.08
    epochs = 1200
    n = len(X)

    for _ in range(epochs):
        grads = [0.0] * 4
        for i in range(n):
            z = sum(weights[j] * X[i][j] for j in range(4))
            pred = 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))
            err = pred - y[i]
            for j in range(4):
                grads[j] += err * X[i][j]
        for j in range(4):
            weights[j] -= lr * (grads[j] / n)

    feature_names = [
        "Intercept (Beta_0)",
        "Rasio_Tagihan_Limit (Beta_1)",
        "Histori_Telat (Beta_2)",
        "Anomali_Diskon (Beta_3)",
    ]
    print("--- Estimasi Parameter Regresi Logistik Biner & Odds Ratio ---")
    print(f"{'VARIABEL PREDIKTOR':<30} | {'KOEFISIEN (BETA)':<18} | {'ODDS RATIO exp(Beta)'}")
    print("-" * 75)
    for name, b in zip(feature_names, weights):
        print(f"{name:<30} | {b:<18.4f} | {math.exp(b):.4f}x lipat risiko")

    correct = 0
    for i, r in enumerate(data):
        z = sum(weights[j] * X[i][j] for j in range(4))
        prob = 1.0 / (1.0 + math.exp(-z))
        r["predicted_prob_late"] = prob
        pred_label = 1 if prob >= 0.5 else 0
        if pred_label == y[i]:
            correct += 1
    print(f"\nAkurasi Klasifikasi Regresi Logistik : {(correct / n) * 100:.2f}%\n")

    print("--- Kurva Survival Kaplan-Meier: Probabilitas Piutang Belum Lunas pada Hari ke-t ---")
    time_horizons = [10, 20, 30, 45, 60, 75]
    n_at_risk = len(data)
    survival_prob = 1.0

    print(f"{'HARI SETELAH JATUH TEMPO (t)':<30} | {'PIUTANG BELUM LUNAS':<22} | {'PROBABILITAS SURVIVAL S(t)'}")
    print("-" * 80)
    for t in time_horizons:
        still_unpaid = sum(1 for r in data if r["days_after_due"] > t)
        s_t = still_unpaid / len(data)
        print(f"Hari ke-{t:<22} | {still_unpaid:<22d} | {s_t:.4f} ({s_t*100:.1f}%)")


if __name__ == "__main__":
    raw_dataset = generate_company_dataset(350)
    clean_dataset = prepare_and_clean_data(raw_dataset)
    multivariate_anomaly_and_hypothesis_test(clean_dataset)
    logistic_regression_and_survival_analysis(clean_dataset)