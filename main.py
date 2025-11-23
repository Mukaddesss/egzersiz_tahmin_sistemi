

import pandas as pd
import numpy as np
import gradio as gr
from functools import lru_cache
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

#  VERİ

df = pd.read_csv("activite_data.csv")

activity_map = {
    0: "Basketball", 1: "Cycling", 2: "Dancing",
    3: "Fidgeting", 4: "HIIT", 5: "Housework",
    6: "Running", 7: "Short Walk", 8: "Sitting",
    9: "Standing", 10: "Swimming", 11: "Tennis",
    12: "Typing", 13: "Walking", 14: "Weight Training",
    15: "Yoga"
}
if "activity_type" not in df.columns:
    df["activity_type"] = df["activity_type_encoded"].map(activity_map)

df["activity_type"] = df["activity_type_encoded"].map(activity_map)


df["activity_type_original"] = df["activity_type"]


df = pd.get_dummies(df, columns=["activity_type"], prefix="act")

features = [
    c for c in df.columns
    if c not in [
        "participant_id",
        "calories_burned",
        "activity_type_encoded",
        "activity_type_original" 
    ]
]


X = df[features]
y = df["calories_burned"]


# MODEL

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- XGBOOST MODEL DOĞRULUK ---")
print(f"MAE  (ortalama mutlak hata): {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   (doğruluk oranı)      : {r2:.4f}")
#print("\nÖrnek gerçek değer :", float(y_test.iloc[0]))
#print("Örnek model tahmini:", float(y_pred[0]))


hr_factor_map = {
    "HIIT": 1.25, "Running": 1.18, "Basketball": 1.15, "Tennis": 1.12,
    "Cycling": 1.10, "Swimming": 1.10, "Weight Training": 1.08,
    "Housework": 1.02, "Short Walk": 0.98, "Walking": 0.98, "Yoga": 0.92,
    "Fidgeting": 0.95, "Standing": 0.95, "Sitting": 0.90, "Typing": 0.90
}

def tahmini_hr(age, base_hr, act_name):
    factor = hr_factor_map.get(act_name, 1.0)
    raw_hr = base_hr * factor
    return min(raw_hr, 220 - age)

def hr_zone(age, hr):
    max_hr = 220 - age
    ratio = hr / max_hr
    if ratio < 0.6:
        return 1, "çok hafif"
    elif ratio < 0.7:
        return 2, "hafif"
    elif ratio < 0.8:
        return 3, "orta"
    elif ratio < 0.9:
        return 4, "yüksek"
    else:
        return 5, "çok yüksek"

def risk_mesaj(zone):
    if zone <= 2:
        return "Genel yüklenme düşük, risk az."
    elif zone == 3:
        return "Orta yoğunlukta, çoğu kişi için genelde güvenli."
    elif zone == 4:
        return "Yüksek yoğunlukta, zorlanma belirtilerine dikkat et."
    else:
        return "Çok yüksek yoğunluk! Bu seviyede spor öncesi doktora danışman önerilir."


#  DATASET FALLBACK (YAŞ+BOY+KİLO+NABIZ)

def dataset_plan(age, gender, height, weight, base_hr, target_cal):
    """
    Sadece gerçekten benzer profiller için dataset kullansın:
    - age farkı ≤ 2
    - height farkı ≤ 1 cm
    - weight farkı ≤ 1 kg
    - nabız farkı ≤ 5 bpm
    - gender birebir eşleşmeli
    + Hedef kaloriden farkı ≤ 50 kcal olmalı, yoksa dataset reddedilir.
    """
    filt = df[
        (df["gender"] == gender) &
        (abs(df["age"] - age) <= 2) &
        (abs(df["height_cm"] - height) <= 1.0) &
        (abs(df["weight_kg"] - weight) <= 1.0) &
        (abs(df["avg_heart_rate"] - base_hr) <= 5.0)
    ]

    if filt.empty:
        return None

    filt = filt.copy()
    filt["cal_diff"] = (filt["calories_burned"] - target_cal).abs()
    row = filt.sort_values("cal_diff").iloc[0]

    if row["cal_diff"] > 50:  
        return None

    activity = row["activity_type_original"]
    duration = float(row["duration_minutes"])
    calories = float(row["calories_burned"])
    hr = float(row["avg_heart_rate"])
    zone, desc = hr_zone(age, hr)

    segment = {
        "activity": activity,
        "duration": duration,
        "calories": calories,
        "hr": hr,
        "zone": zone,
        "zone_desc": desc
    }

    plan = {
        "source": "DATASET",
        "segments": [segment],
        "total_cal": calories,
        "total_duration": duration,
        "max_zone": zone,
        "score": 0.0
    }
    return plan


@lru_cache(maxsize=None)
def predict_block(age, gender, height, weight, base_hr, duration, activity_enc):
    activity_name = activity_map[int(activity_enc)]
    hr = tahmini_hr(age, base_hr, activity_name)
    bmi = weight / ((height / 100) ** 2)

    sample = {col: 0 for col in features}  
    sample["age"] = age
    sample["gender"] = gender
    sample["height_cm"] = height
    sample["weight_kg"] = weight
    sample["duration_minutes"] = duration
    sample["avg_heart_rate"] = hr
    sample["bmi"] = bmi

    onehot_col = f"act_{activity_name}"
    if onehot_col in sample:
        sample[onehot_col] = 1

    df_sample = pd.DataFrame([sample])
    cal = float(model.predict(df_sample)[0])

    zone, zone_desc = hr_zone(age, hr)
    return cal, hr, zone, zone_desc, activity_name


def get_duration_blocks_for_activity(age, gender, height, weight, base_hr, enc, max_total_duration=90):
    """
    Dinamik süre blokları:
    - 10 dk tahmini kaloriyi hesapla
    - düşük yakıyorsa daha uzun,
      yüksek yakıyorsa daha kısa bloklar.
    """
    cal_10, _, _, _, _ = predict_block(age, gender, height, weight, base_hr, 10, enc)

    if cal_10 < 20:
        candidate = [20, 30, 40, 50, 60, 90]
    elif cal_10 < 40:
        candidate = [10, 20, 30, 40, 50, 60]
    else:
        candidate = [10, 20, 30, 40]

    return [d for d in candidate if d <= max_total_duration] or [10]

def evaluate_plan_score(total_cal, target_cal, total_dur, avg_zone):
    """
    Skor ne kadar küçükse plan o kadar iyi:
    - hedef kaloriden sapma (en ağır)
    - yüksek zone için ceza
    - 60 dk üstü süre için hafif ceza
    """
    cal_diff = abs(total_cal - target_cal)
    zone_penalty = max(0, avg_zone - 2) * 5   # yumuşak ceza
    duration_penalty = max(0, total_dur - 60) # her ekstra dk 1 puan
    return cal_diff + zone_penalty + duration_penalty


# TEK AKTİVİTELİ MODEL PLANI

def single_model_plan(age, gender, height, weight, base_hr, target_cal, max_total_duration=90):
    aktiviteler = df[["activity_type_original", "activity_type_encoded"]].drop_duplicates()
    aktiviteler = aktiviteler.rename(columns={"activity_type_original": "activity_type"})

    best_plan = None
    best_score = float("inf")

    for _, row in aktiviteler.iterrows():
        enc = int(row["activity_type_encoded"])
        durations = get_duration_blocks_for_activity(age, gender, height, weight, base_hr, enc, max_total_duration)

        for dur in durations:
            cal, hr, zone, desc, act_name = predict_block(age, gender, height, weight, base_hr, dur, enc)
            total_cal = cal
            total_dur = dur
            avg_zone = zone

            score = evaluate_plan_score(total_cal, target_cal, total_dur, avg_zone)

            if score < best_score:
                best_score = score
                best_plan = {
                    "source": "MODEL_SINGLE",
                    "segments": [{
                        "activity": act_name,
                        "duration": dur,
                        "calories": cal,
                        "hr": hr,
                        "zone": zone,
                        "zone_desc": desc
                    }],
                    "total_cal": total_cal,
                    "total_duration": total_dur,
                    "max_zone": zone,
                    "score": score
                }

    return best_plan


# ÇOKLU AKTİVİTE MODEL PLANI

def multi_model_plan(age, gender, height, weight, base_hr, target_cal,
                     max_total_duration=90, max_segments=6, tol_improve=5, tol_cal=10):
    """
    Greedy bir şekilde:
    - Birden fazla aktivite bloğunu toplayarak hedef kaloriye yaklaşır.
    - Aynı aktiviteyi tekrar seçebilir ama çeşitlilik için küçük ceza vardır.
    """
    aktiviteler = df[["activity_type_original", "activity_type_encoded"]].drop_duplicates()
    aktiviteler = aktiviteler.rename(columns={"activity_type_original": "activity_type"})


    
    candidates = []
    for _, row in aktiviteler.iterrows():
        enc = int(row["activity_type_encoded"])
        durations = get_duration_blocks_for_activity(age, gender, height, weight, base_hr, enc, max_total_duration)

        for dur in durations:
            cal, hr, zone, desc, act_name = predict_block(age, gender, height, weight, base_hr, dur, enc)
            candidates.append({
                "activity": act_name,
                "duration": dur,
                "calories": cal,
                "hr": hr,
                "zone": zone,
                "zone_desc": desc
            })

    if not candidates:
        return None

    segments = []
    total_cal = 0.0
    total_dur = 0.0
    avg_zone = 1.0
    current_score = evaluate_plan_score(total_cal, target_cal, total_dur, avg_zone)

    for _ in range(max_segments):
        best_seg = None
        best_seg_score = None
        best_new_cal = None
        best_new_dur = None
        best_new_avg_zone = None

        used_acts = {seg["activity"] for seg in segments}

        for cand in candidates:
            if total_dur + cand["duration"] > max_total_duration:
                continue

            new_total_cal = total_cal + cand["calories"]
            new_total_dur = total_dur + cand["duration"]

            if len(segments) == 0:
                new_avg_zone = cand["zone"]
            else:
                new_avg_zone = (avg_zone * len(segments) + cand["zone"]) / (len(segments) + 1)

            base_score = evaluate_plan_score(new_total_cal, target_cal, new_total_dur, new_avg_zone)

            
            variety_penalty = 5 if cand["activity"] in used_acts else 0
            new_score = base_score + variety_penalty

            if best_seg is None or new_score < best_seg_score:
                best_seg = cand
                best_seg_score = new_score
                best_new_cal = new_total_cal
                best_new_dur = new_total_dur
                best_new_avg_zone = new_avg_zone

        if best_seg is None:
            break

        
        if best_seg_score > current_score + tol_improve and total_cal > 0:
            break

        
        segments.append(best_seg)
        total_cal = best_new_cal
        total_dur = best_new_dur
        avg_zone = best_new_avg_zone
        current_score = best_seg_score

       
        if abs(total_cal - target_cal) <= tol_cal:
            break

    if not segments:
        return None

    max_zone = max(seg["zone"] for seg in segments)

    return {
        "source": "MODEL_MULTI",
        "segments": segments,
        "total_cal": total_cal,
        "total_duration": total_dur,
        "max_zone": max_zone,
        "score": current_score
    }


def aktivite_oner(age, gender, height, weight, base_hr, target_cal):
    plan_ds = dataset_plan(age, gender, height, weight, base_hr, target_cal)
    if plan_ds is not None:
        return plan_ds

   
    single = single_model_plan(age, gender, height, weight, base_hr, target_cal)

    
    multi = multi_model_plan(age, gender, height, weight, base_hr, target_cal)

    if multi is None:
        return single

    if multi["score"] <= single["score"] * 1.10:
        return multi
    else:
        return single


def format_output(plan, target_cal):
    source = plan["source"]
    total_cal = plan["total_cal"]
    total_dur = plan["total_duration"]
    max_zone = plan["max_zone"]
    segments = plan["segments"]

    out = ""
    out += "================= ÖNERİLEN ANTRENMAN PLANI =================\n"
    out += f"Hedef Kalori      : {target_cal:.2f} kcal\n"
    out += f"Tahmini Toplam    : {total_cal:.2f} kcal\n"
    out += f"Toplam Süre       : {total_dur:.1f} dk\n"
    out += f"Plan Kaynağı      : {source}\n\n"

    out += "Bölümler:\n"
    for i, seg in enumerate(segments, 1):
        out += f"{i}) Aktivite : {seg['activity']}\n"
        out += f"   Süre     : {seg['duration']:.1f} dk\n"
        out += f"   Kalori   : {seg['calories']:.2f} kcal\n"
        out += f"   Nabız    : {seg['hr']} bpm\n"
        out += f"   Yoğunluk : {seg['zone_desc']}\n\n"

    out += "Genel Risk Değerlendirmesi:\n"
    out += risk_mesaj(max_zone) + "\n"

    if max_zone >= 4:
        out += (
            "\n⚠ Uyarı: Bu planda yüksek nabız bölgeleri içeriyor.\n"
            "Yorgunluk, göğüs ağrısı, baş dönmesi gibi belirtiler hissedersen hemen durmalısın.\n"
        )
    elif max_zone == 3:
        out += "\nNot: Orta yoğunluklu bir plan. Rahatsızlık hissetmezsen genelde güvenlidir.\n"
    else:
        out += "\nPlan düşük/orta yoğunluklu, genel olarak güvenli.\n"

    return out


# GRADIO FONKSİYONU

def generate_plan(age, gender, height, weight, base_hr, hedef_kalori):
    plan = aktivite_oner(
        age=int(age),
        gender=int(gender),
        height=float(height),
        weight=float(weight),
        base_hr=float(base_hr),
        target_cal=float(hedef_kalori)
    )
    return format_output(plan, float(hedef_kalori))


with gr.Blocks(title="Akıllı Kalori ve Aktivite Planlama") as demo:

    gr.Markdown("## 🏋️ Akıllı Aktivite & Kalori Tahmin Sistemi")

    with gr.Row():
        age = gr.Number(label="Yaş", value=25, minimum=5, maximum=100)
        gender = gr.Dropdown(["0", "1"], label="Cinsiyet (0/1)", value="0")

    with gr.Row():
        height = gr.Number(label="Boy (cm)", value=170, minimum=50, maximum=250)
        weight = gr.Number(label="Kilo (kg)", value=65, minimum=10, maximum=200)

    with gr.Row():
        base_hr = gr.Number(label="Ortalama Nabız (bpm)", value=130, minimum=40, maximum=220)
        hedef_kalori = gr.Number(label="Hedef Kalori (kcal)", value=300, minimum=1)

    generate_button = gr.Button("Plan Oluştur")

    output_box = gr.Textbox(
        label="Önerilen Antrenman Planı",
        lines=20,
        interactive=False
    )

    generate_button.click(
        fn=generate_plan,
        inputs=[age, gender, height, weight, base_hr, hedef_kalori],
        outputs=output_box
    )

demo.launch()