# %%
# 필요 라이브러리
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr

# 모듈화된 함수 임포트
from similarity import match_raman_spectrum2

# %%
st.title("Unknown Raman Spectrum Analyzer")

st.markdown("""
이 프로그램은 Unknown sample의 240~3200 cm⁻¹ 영역의 Raman spectrum을 
우리의 database와 비교 분석하여 미지의 샘플을 식별합니다.
""")

st.subheader("CSV 파일 업로드")
st.write("""
PNG spectrum image는 먼저 CSV 파일로 변환되어야 합니다.
CSV 변환은 PlotDigitizer.com 을 이용해주세요.
""")

# %%
# 업로드 csv/xlsx 읽기
input_file = st.file_uploader(
    "파일 업로드",
    type=["csv"]
)

# 분석 버튼 클릭 시 실행되는 블록
if st.button("Analyze"):
    if input_file is None:
        st.warning("업로드할 CSV 파일을 선택해주세요.")
    else:
        # 1. 진행 상태 표시
        with st.spinner("데이터 전처리 및 분석 진행 중..."):
            uploaded_data = pd.read_csv(input_file)
            
            # 🔥 핵심 수정 위치: 함수에서 (결과 리스트, fig 객체) 두 개를 받도록 수정합니다.
            results, fig = match_raman_spectrum2(
                target_raw=uploaded_data,
                db_csv_path='data/lib_snv.csv',
                method='correlation',
                top_k=5,
                plot_results=True  # 필요시 시각화 옵션 플래그 전달
            )

        # 2. 완료 메시지 및 결과 출력
        st.success("분석 완료!")

        # 결과 표출 (표 형태로 출력)
        st.subheader("📊 분석 결과")
        if results:
            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True)
        else:
            st.info("일치하는 결과가 없습니다.")

        # 3. 그래프 출력 (fig가 존재할 경우에만 출력)
        if fig is not None:
            st.subheader("📈 스펙트럼 매칭 그래프")
            st.pyplot(fig)