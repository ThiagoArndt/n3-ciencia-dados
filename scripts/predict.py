import pandas as pd
import joblib
import sys

modelo = joblib.load('../modelo_final.pkl')
scaler = joblib.load('../scaler.pkl')

novo_paciente = {
    'age': int(sys.argv[1]) if len(sys.argv) > 1 else 55,
    'sex': int(sys.argv[2]) if len(sys.argv) > 2 else 1,
    'chest_pain_type': int(sys.argv[3]) if len(sys.argv) > 3 else 2,
    'resting_bp': int(sys.argv[4]) if len(sys.argv) > 4 else 140,
    'cholesterol': int(sys.argv[5]) if len(sys.argv) > 5 else 260,
    'fasting_bs': int(sys.argv[6]) if len(sys.argv) > 6 else 0,
    'resting_ecg': int(sys.argv[7]) if len(sys.argv) > 7 else 0,
    'max_hr': int(sys.argv[8]) if len(sys.argv) > 8 else 150,
    'exercise_angina': int(sys.argv[9]) if len(sys.argv) > 9 else 1,
    'oldpeak': float(sys.argv[10]) if len(sys.argv) > 10 else 2.5,
    'st_slope': int(sys.argv[11]) if len(sys.argv) > 11 else 1,
    'num_vessels': int(sys.argv[12]) if len(sys.argv) > 12 else 1,
    'thalassemia': int(sys.argv[13]) if len(sys.argv) > 13 else 3
}

df = pd.DataFrame([novo_paciente])
df_scaled = scaler.transform(df)
previsao = modelo.predict(df_scaled)
probabilidade = modelo.predict_proba(df_scaled)

print(f"Previsão: {'POSITIVO para doença cardíaca' if previsao[0] == 1 else 'NEGATIVO para doença cardíaca'}")
print(f"Probabilidade de NÃO ter doença: {probabilidade[0][0]:.2%}")
print(f"Probabilidade de TER doença: {probabilidade[0][1]:.2%}")

