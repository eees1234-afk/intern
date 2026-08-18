import pandas as pd

df_snv = pd.read_csv("data/lib_snv.csv", header=None)
df_als = pd.read_csv("data/lib_als.csv", skiprows=3, header=None)

intensity = df_snv.iloc[:, 0].values
x = df_als.iloc[:, 0].values

df_result = pd.DataFrame({'wavenumber': x, 'intensity': intensity})
df_result.to_csv("data/test3.csv", index=False)

print("test3.csv 파일이 성공적으로 생성되었습니다.")

exit()  # 스크립트 종료