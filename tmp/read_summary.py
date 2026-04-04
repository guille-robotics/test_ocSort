import csv

with open('videos_salida/v3_resumen_batch.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"{'Video':<10} {'FPS_Src':>7} {'Frames':>7} {'IDs':>5} {'MaxTrk':>7} {'AvgDet':>7} {'ProcFPS':>8} {'%Alerta':>8}")
print('-'*65)
for r in rows:
    print(f"{r['Video']:<10} {r['FPS_Fuente']:>7} {r['Frames_Procesados']:>7} {r['IDs_Unicos']:>5} {r['Max_Tracks_Simultaneos']:>7} {r['Avg_Detecciones_Frame']:>7} {r['FPS_Procesamiento']:>8} {r['Pct_Frames_Alerta']:>8}")
