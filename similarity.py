import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def baseline_als(y, lam=1e6, p=0.001, niter=10):
    """AsLS (Asymmetric Least Squares) 베이스라인 추정 함수"""
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = diags([w], [0])
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def match_raman_spectrum2(
    target_raw,
    db_csv_path,
    method='correlation',
    top_k=3,
    lam=1e6,
    p=0.001,
    plot_results=True,
):
    """Target 스펙트럼에 [내삽 -> AsLS 베이스라인 제거 -> SNV] 전처리를 순차적으로 적용한 후

    DB 스펙트럼과 유사도를 매칭하고 시각화합니다.
    """
    # -----------------------------------------------------------------
    # 1. DB 데이터 로드
    # -----------------------------------------------------------------
    db_df = pd.read_csv(db_csv_path, skiprows=[0, 2])

    # 0번 열: DB 파수(Wavenumber), 1번 열부터: 물질 스펙트럼들
    db_wn = db_df.iloc[:, 0].values.astype(float)
    db_labels = db_df.columns[1:].tolist()

    # (데이터개수, 물질수) -> (물질수, 데이터개수) 구조로 전치
    db_spectra = db_df.iloc[:, 1:].values.T.astype(float)

  # -----------------------------------------------------------------
    # 2. Target 데이터 입력 파싱 (유효한 데이터 2열 자동 추출)
    # -----------------------------------------------------------------
    if isinstance(target_raw, pd.DataFrame):
        # 숫자로 변환 불가능한 문자열 정보칸이나 빈 열 등을 제외하기 위해 숫자형(numeric) 데이터만 필터링
        df_numeric = target_raw.select_dtypes(include=[np.number])
        
        # 만약 변환이 안 된 문자열 형태의 숫자가 있다면 숫자형으로 강제 변환 후 결측치(NaN)가 많은 열 제거
        if df_numeric.shape[1] < 2:
            df_numeric = target_raw.apply(pd.to_numeric, errors='coerce')
        
        # 결측치(NaN)가 없는(완전히 꽉 채워진) 열만 추출
        valid_cols = df_numeric.dropna(axis=1, how='any')
        
        # 꽉 채워진 열이 2개 이상이면 가장 앞쪽 2개 열 선택
        if valid_cols.shape[1] >= 2:
            target_arr = valid_cols.iloc[:, :2].values
        else:
            # 완전히 채워진 열이 부족할 경우 결측치가 가장 적은 유효 열 2개 선택
            valid_col_indices = df_numeric.notna().sum().nlargest(2).index
            target_arr = df_numeric[valid_col_indices].values
    else:
        target_arr = np.asarray(target_raw)

    # 행/열 방향 차원 처리 및 2개 열 추출
    if target_arr.ndim == 2:
        # (N, M) 형태일 때 열(Column)이 2개 이상인 경우
        if target_arr.shape[1] >= 2:
            target_wn = target_arr[:, 0].astype(float)
            target_signal = target_arr[:, 1].astype(float)
        # (2, N) 형태로 전치되어 들어온 경우
        elif target_arr.shape[0] >= 2:
            target_wn = target_arr[0, :].astype(float)
            target_signal = target_arr[1, :].astype(float)
        else:
            raise ValueError("유효한 데이터 열을 2개 이상 찾을 수 없습니다.")
    else:
        raise ValueError("target_raw는 2차원 데이터 형태여야 합니다.")

    
    # -----------------------------------------------------------------
    # 3. 자동 크롭(Cropping) 영역 계산 및 1 cm⁻¹ 정수 파수 Grid 생성
    # -----------------------------------------------------------------
    crop_min = max(db_wn.min(), target_wn.min())
    crop_max = min(db_wn.max(), target_wn.max())

    start_wn = int(np.ceil(crop_min))
    end_wn = int(np.floor(crop_max))

    if start_wn >= end_wn:
        raise ValueError(
            "DB와 Target 데이터 간에 겹치는 파수 영역이 없습니다."
        )

    common_wn_grid = np.arange(start_wn, end_wn + 1)

    # -----------------------------------------------------------------
    # 4. Target 전처리 Pipeline (내삽 -> AsLS -> SNV)
    # -----------------------------------------------------------------
    # Step 4-1) 정수 파수 축으로 내삽
    f_target = interp1d(
        target_wn, target_signal, bounds_error=False, fill_value="extrapolate"
    )
    target_interp = f_target(common_wn_grid)

    # Step 4-2) AsLS 베이스라인 제거
    baseline = baseline_als(target_interp, lam=lam, p=p)
    target_corrected = target_interp - baseline

    # Step 4-3) SNV (Standard Normal Variate) 적용
    std_val = np.std(target_corrected)
    if std_val == 0:
        target_snv = target_corrected - np.mean(target_corrected)
    else:
        target_snv = (target_corrected - np.mean(target_corrected)) / std_val

    # -----------------------------------------------------------------
    # 5. DB 데이터 크롭 (공통 정수 파수 영역으로 동기화)
    # -----------------------------------------------------------------
    f_db = interp1d(
        db_wn,
        db_spectra,
        axis=1,
        bounds_error=False,
        fill_value="extrapolate",
    )
    db_spectra_cropped = f_db(common_wn_grid)

    # -----------------------------------------------------------------
    # 6. 유사도 계산
    # -----------------------------------------------------------------
    results = []

    for i, db_spec in enumerate(db_spectra_cropped):
        if method == "cosine":
            score = 1 - cosine(target_snv, db_spec)
        elif method == "euclidean":
            score = euclidean(target_snv, db_spec)
        elif method == "correlation":
            score, _ = pearsonr(target_snv, db_spec)
        else:
            raise ValueError(
                "Support methods: 'cosine', 'euclidean', 'correlation'"
            )

        results.append(
            {
                "label": db_labels[i],
                "score": float(score),
                "spectrum": db_spec,
            }
        )

    # -----------------------------------------------------------------
    # 7. 정렬 및 상위 결과 반환
    # -----------------------------------------------------------------
    if method == "euclidean":
        results.sort(key=lambda x: x["score"])
    else:
        results.sort(key=lambda x: x["score"], reverse=True)

    top_results = results[:top_k]

    # -----------------------------------------------------------------
    # 8. 시각화
    # -----------------------------------------------------------------
    if plot_results and top_results:
        num_plots = len(top_results)
        fig, axes = plt.subplots(
            num_plots, 1, figsize=(10, 3.8 * num_plots), sharex=True
        )

        if num_plots == 1:
            axes = [axes]

        for i, (ax, result) in enumerate(zip(axes, top_results)):
            ax.plot(
                common_wn_grid,
                target_snv,
                label="Target (AsLS + SNV)",
                color="black",
                linewidth=1.5,
            )
            ax.plot(
                common_wn_grid,
                result["spectrum"],
                label=f"Match #{i+1}: {result['label']}",
                color="red",
                linestyle="--",
                alpha=0.85,
            )

            ax.set_title(
                f"Rank {i+1} - Method: {method.capitalize()}, Score:"
                f" {result['score']:.4f}"
            )
            ax.set_ylabel("Intensity (SNV)")
            ax.legend(loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.6)

        axes[-1].set_xlabel(r"Wavenumber ($cm^{-1}$)")
        plt.tight_layout()
        # plt.show()

# -----------------------------------------------------------------
    # 최종 반환
    # -----------------------------------------------------------------
    results = [{"label": r["label"], "score": r["score"]} for r in top_results]

    # 🔥 3. results와 함께 fig 객체도 같이 반환합니다. (기존: return [ ... ])
    return results, fig