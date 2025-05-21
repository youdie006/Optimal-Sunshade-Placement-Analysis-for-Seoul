import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import platform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE  # PCA 대신 TSNE 사용
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import matplotlib.font_manager as fm
import os
import warnings

warnings.filterwarnings('ignore')

# macOS용 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Windows
    plt.rcParams['font.family'] = 'Malgun Gothic'

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
print("분석을 시작합니다...")

# 1. 데이터 로드 및 전처리
data = pd.read_csv('Seoul_Dongs_Temperature_Features_202408132.csv', encoding='UTF-8')

# 행정동 경계 데이터 로드 (GeoJSON 또는 Shapefile)
try:
    # GeoJSON 파일 경로 (파일명을 실제 파일명으로, 확장자도 맞게 수정하세요)
    seoul_geo_path = 'hangjeongdong_서울특별시.geojson'  # 또는 .shp 등 실제 파일에 맞게 수정
    seoul_geo = gpd.read_file(seoul_geo_path, encoding='UTF-8')
    has_geo_data = True
    print(f"행정동 경계 데이터 로드 완료: {seoul_geo.shape[0]}개 행정동")
except Exception as e:
    has_geo_data = False
    print(f"행정동 경계 데이터 로드 실패: {e}")
    print("지도 시각화는 건너뜁니다.")

# 결측치 확인 및 처리
print(f"데이터 크기: {data.shape}")
print(f"결측치 현황:\n{data.isnull().sum()}")

# NaN 값을 가진 행 수 확인
missing_rows = data[data.isnull().any(axis=1)]
print(f"결측치가 있는 행: {len(missing_rows)}개")

# 필요한 경우 결측치 처리
data = data.dropna(subset=['mean', 'max'])  # hot_area_ratio를 사용하지 않으므로 제외
print(f"결측치 처리 후 데이터 크기: {data.shape}")

# 분석에 필요한 특성 선택 - hot_area_ratio 대신 p50 추가
features = ['mean', 'max', 'min', 'p50', 'p90', 'stdDev']  # p50 추가, hot_area_ratio 제외
X = data[features].copy()

# 온도 변동 지수 추가 (90백분위수 - 50백분위수)
data['temp_variation'] = data['p90'] - data['p50']

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# 2. K-means 클러스터링
# 최적의 클러스터 수 결정 (엘보우 메서드)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o', color='#4A90E2', linewidth=2)
plt.title('최적 클러스터 수 결정 (Elbow Method)', fontsize=16)
plt.xlabel('클러스터 수', fontsize=14)
plt.ylabel('Inertia', fontsize=14)
plt.xticks(range(1, 11))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=300)
plt.close()

# K-means 클러스터링 적용 (k=5 가정)
k = 5  # 엘보우 메서드 결과에 따라 조정 가능
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 클러스터 결과를 원본 데이터에 추가
data['cluster'] = clusters

# 3. 개선된 그늘막 필요 지수 계산
# 가중치: 최대 온도 50%, 평균 온도 30%, 온도 변동 지수 20%
data['shade_need_index'] = (data['max'] * 0.5 +
                            data['mean'] * 0.3 +
                            data['temp_variation'] * 0.2)  # hot_area_ratio 대신 temp_variation 사용

# 클러스터별 그늘막 필요 지수 평균
cluster_shade_need = data.groupby('cluster')['shade_need_index'].mean().sort_values(ascending=False)
hottest_cluster = cluster_shade_need.index[0]

# 4. 상위 3개 행정동 선택
top_3_dongs = data[data['cluster'] == hottest_cluster].sort_values('shade_need_index', ascending=False).head(3)
print("\n그늘막 설치 우선순위 상위 3개 행정동:")
print(
    top_3_dongs[['ADM_NM', 'max', 'mean', 'temp_variation', 'shade_need_index']])  # hot_area_ratio 대신 temp_variation 표시

# 5. 시각화 - 클러스터별 열 특성
plt.figure(figsize=(16, 10))
heat_features = ['mean', 'max', 'temp_variation', 'shade_need_index']  # hot_area_ratio 대신 temp_variation 표시

for i, feature in enumerate(heat_features):
    plt.subplot(2, 2, i + 1)

    # 더 화려한 색상 팔레트 사용
    sns.boxplot(x='cluster', y=feature, data=data,
                palette=sns.color_palette("viridis", k), width=0.6)

    # 상위 3개 행정동 강조 표시
    if len(top_3_dongs) > 0:
        plt.scatter(
            [hottest_cluster] * len(top_3_dongs),
            top_3_dongs[feature],
            color='red', marker='*', s=200, label='상위 3개 행정동' if i == 0 else ""
        )

    # 그래프 스타일 개선
    plt.title(f'클러스터별 {feature} 분포', fontsize=14)
    plt.xlabel('클러스터', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')

    if i == 0:
        plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('cluster_heat_features.png', dpi=300)
plt.close()

# 6. t-SNE 클러스터링 시각화 (PCA 대신 t-SNE 사용)
# 온도 변동 지수를 포함한 특성 목록 생성
features_with_variation = features + ['temp_variation']
X_with_variation = data[features_with_variation].copy()

# 모든 특성 스케일링
scaler_with_variation = StandardScaler()
X_scaled_with_variation = scaler_with_variation.fit_transform(X_with_variation)

# t-SNE 적용
print("t-SNE를 활용한 클러스터 시각화 생성 중...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)
X_tsne = tsne.fit_transform(X_scaled_with_variation)

# 시각화
plt.figure(figsize=(14, 12))

# 배경 스타일 설정
plt.grid(True, linestyle='--', alpha=0.7)

# 클러스터별 마커와 색상 정의
cluster_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
markers = ['o', 's', '^', 'D', 'P']

# 클러스터별 데이터 포인트 시각화
for i in range(k):
    cluster_points = X_tsne[clusters == i]
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        s=80,
        c=cluster_colors[i],
        marker=markers[i],
        alpha=0.8,
        edgecolor='white',
        linewidth=0.5,
        label=f'클러스터 {i}'
    )

# 상위 3개 행정동 강조 표시
top_3_indices = top_3_dongs.index
plt.scatter(
    X_tsne[top_3_indices, 0],
    X_tsne[top_3_indices, 1],
    s=250,
    marker='*',
    color='red',
    edgecolor='yellow',
    linewidth=2.5,
    label='상위 3개 행정동',
    zorder=5
)

# 상위 행정동 라벨 추가
for i, idx in enumerate(top_3_indices):
    plt.annotate(
        f"{data.loc[idx, 'ADM_NM']}",
        (X_tsne[idx, 0] + 0.5, X_tsne[idx, 1] + 0.5),
        fontsize=14,
        weight='bold',
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="black", alpha=0.8),
        zorder=6
    )

# 그래프 스타일 개선
plt.title('서울시 행정동 온도 특성 클러스터링 (t-SNE)', fontsize=20, fontweight='bold')
plt.xlabel('t-SNE 특성 1', fontsize=16)
plt.ylabel('t-SNE 특성 2', fontsize=16)

# 향상된 범례
plt.legend(
    fontsize=14,
    markerscale=1.5,
    frameon=True,
    framealpha=0.9,
    facecolor='white',
    edgecolor='gray',
    shadow=True,
    loc='best'
)

# 클러스터 특성 설명 텍스트 추가
description = (
    "t-SNE 시각화 특징:\n"
    "• 고차원 데이터의 클러스터 구조를 2D로 표현\n"
    "• 유사한 온도 특성을 가진 행정동은 가까이 위치\n"
    "• 별표(*)는 그늘막 설치 우선순위가 가장 높은 행정동\n"
    "• 분석에 사용된 특성: 평균·최대·최소 온도, 백분위수(p50, p90), 온도 변동 지수"
)

# 설명 텍스트 박스 추가
plt.figtext(
    0.5, 0.02,
    description,
    ha='center',
    fontsize=14,
    bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9)
)

# 저장 옵션 개선
plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig('seoul_temperature_tsne.png', dpi=300, bbox_inches='tight')
plt.close()

print("t-SNE 클러스터 시각화 완료: seoul_temperature_tsne.png 저장됨")

# 7. 히트맵 - 클러스터 중심값 특성 비교
plt.figure(figsize=(12, 8))
# 클러스터 중심값에 온도 변동 지수 추가
extended_centers = np.zeros((k, len(features) + 1))
extended_centers[:, :-1] = kmeans.cluster_centers_
# 각 클러스터의 평균 온도 변동 지수 계산
for i in range(k):
    extended_centers[i, -1] = data[data['cluster'] == i]['temp_variation'].mean()

# 역변환을 위한 스케일러 선택
cluster_centers_df = pd.DataFrame(scaler_with_variation.inverse_transform(
    np.hstack((kmeans.cluster_centers_, np.zeros((k, 1))))
), columns=features_with_variation)

# 온도 변동 지수 값 직접 설정 (스케일러가 각 클러스터 평균값으로 변환해주지 않으므로)
for i in range(k):
    cluster_centers_df.loc[i, 'temp_variation'] = data[data['cluster'] == i]['temp_variation'].mean()

# 클러스터별 그늘막 필요 지수 추가
cluster_centers_df['shade_need_index'] = cluster_shade_need.values

# 히트맵에 표시할 열 선택
heatmap_columns = features + ['temp_variation', 'shade_need_index']
cluster_centers_df = cluster_centers_df[heatmap_columns]

# 시각화를 위한 열 이름 변경
pretty_names = {
    'mean': '평균 온도',
    'max': '최대 온도',
    'min': '최소 온도',
    'p50': '중앙값 온도',
    'p90': '90백분위 온도',
    'temp_variation': '온도 변동 지수',
    'stdDev': '표준편차',
    'shade_need_index': '그늘막 필요 지수'
}
cluster_centers_df.columns = [pretty_names.get(col, col) for col in cluster_centers_df.columns]

# 히트맵 그리기
sns.heatmap(
    cluster_centers_df.T,
    annot=True,
    cmap='RdYlBu_r',
    fmt='.2f',
    linewidths=.5,
    cbar_kws={"label": "값"}
)

plt.title('클러스터별 중심값 특성 비교', fontsize=16)
plt.xlabel('클러스터', fontsize=14)
plt.ylabel('특성', fontsize=14)
plt.tight_layout()
plt.savefig('cluster_centers_heatmap.png', dpi=300)
plt.close()

# 8. 클러스터별 그늘막 필요 지수 비교 시각화
plt.figure(figsize=(12, 6))

# 클러스터별 그늘막 필요 지수 평균 막대 그래프
bars = plt.bar(
    cluster_shade_need.index,
    cluster_shade_need.values,
    color=cluster_colors,
    width=0.6,
    edgecolor='black',
    linewidth=1.5
)

# 가장 더운 클러스터 강조
bars[list(cluster_shade_need.index).index(hottest_cluster)].set_edgecolor('red')
bars[list(cluster_shade_need.index).index(hottest_cluster)].set_linewidth(3)

# 클러스터별 행정동 수 표시
for i, (cluster, value) in enumerate(cluster_shade_need.items()):
    count = len(data[data['cluster'] == cluster])
    plt.text(
        i,
        value + 0.5,
        f'n={count}',
        ha='center',
        va='bottom',
        fontsize=12,
        weight='bold' if cluster == hottest_cluster else 'normal'
    )

# 그래프 스타일 개선
plt.title('클러스터별 평균 그늘막 필요 지수', fontsize=16)
plt.xlabel('클러스터', fontsize=14)
plt.ylabel('그늘막 필요 지수', fontsize=14)
plt.xticks(range(k))
plt.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('cluster_shade_need_index.png', dpi=300)
plt.close()

# 9. 상위 3개 행정동 특성 비교
plt.figure(figsize=(14, 7))

# 비교할 특성 - hot_area_ratio 대신 temp_variation 사용
compare_features = ['max', 'mean', 'temp_variation', 'shade_need_index']
positions = np.arange(len(compare_features))
width = 0.25  # 막대 너비

# 각 행정동별 막대 그래프
for i, (idx, row) in enumerate(top_3_dongs.iterrows()):
    values = [row[feature] for feature in compare_features]
    plt.bar(
        positions + i * width - width,
        values,
        width=width,
        label=f"{row['ADM_NM']}",
        color=plt.cm.Set2(i),
        edgecolor='black',
        linewidth=1
    )

# 그래프 스타일 개선
plt.title('상위 3개 행정동 온도 특성 비교', fontsize=16)
plt.xticks(positions, [pretty_names.get(feat, feat) for feat in compare_features], fontsize=12)
plt.ylabel('값', fontsize=14)
plt.legend(title='행정동', fontsize=12, loc='upper right')
plt.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('top_3_dongs_comparison.png', dpi=300)
plt.close()

# 서울시 행정동별 그늘막 필요 지수 시각화 - 서울 지역만 정확히 표시
if has_geo_data:
    try:
        print("서울시 행정동별 그늘막 필요 지수 시각화 시작...")

        # 행정동 코드 열 확인
        geo_key = 'adm_cd8' if 'adm_cd8' in seoul_geo.columns else 'adm_cd'
        data_key = 'ADM_CD'

        # 데이터 준비
        seoul_geo_copy = seoul_geo.copy()
        data_copy = data.copy()

        # 행정동 코드를 문자열로 변환
        seoul_geo_copy[geo_key] = seoul_geo_copy[geo_key].astype(str)
        data_copy[data_key] = data_copy[data_key].astype(str)

        # 직접 매핑 실행
        cluster_dict = dict(zip(data_copy[data_key], data_copy['cluster']))
        shade_dict = dict(zip(data_copy[data_key], data_copy['shade_need_index']))
        temp_var_dict = dict(zip(data_copy[data_key], data_copy['temp_variation']))  # 온도 변동 지수 추가

        seoul_geo_copy['cluster_val'] = seoul_geo_copy[geo_key].map(cluster_dict)
        seoul_geo_copy['shade_need_index'] = seoul_geo_copy[geo_key].map(shade_dict)
        seoul_geo_copy['temp_variation'] = seoul_geo_copy[geo_key].map(temp_var_dict)  # 온도 변동 지수 추가

        # 매핑 결과 확인
        mapped_count = seoul_geo_copy['cluster_val'].notna().sum()
        print(f"매핑된 행정동 수: {mapped_count}/{len(seoul_geo_copy)}개")

        # ======= 서울만 필터링 =======
        # 방법 1: 맵핑된 행정동(서울)만 필터링하여 별도의 GeoDataFrame 생성
        seoul_only = seoul_geo_copy[seoul_geo_copy['cluster_val'].notna()].copy()
        print(f"서울 행정동으로 필터링된 데이터: {len(seoul_only)}개")

        # 서울 지역만 표시하는 향상된 시각화
        fig, ax = plt.subplots(figsize=(15, 15))

        # ======= 중요: 서울만 포함한 GeoDataFrame 사용 =======
        # 기존 seoul_geo_copy 대신 seoul_only 사용

        # 시군구 경계선 먼저 그리기 (더 두껍게)
        if 'sgg' in seoul_only.columns:
            try:
                sgg_boundaries = seoul_only.dissolve(by='sgg').boundary
                sgg_boundaries.plot(
                    ax=ax,
                    color='black',
                    linewidth=1.5,
                    alpha=0.7
                )
            except Exception as e:
                print(f"구 경계선 추가 중 오류: {e}")

        # 클러스터별 색상 정의
        cluster_colors = ['#0d47a1', '#1976d2', '#f5f5f5', '#fdd835', '#ff9800']

        # 클러스터별 시각화
        for i in range(k):
            cluster_geo = seoul_only[seoul_only['cluster_val'] == i]
            if len(cluster_geo) > 0:
                cluster_geo.plot(
                    ax=ax,
                    color=cluster_colors[i],
                    edgecolor='grey',
                    linewidth=0.8,
                    alpha=0.7
                )

        # 상위 3개 행정동 강조
        top_codes = top_3_dongs[data_key].astype(str).tolist()
        top_geo = seoul_only[seoul_only[geo_key].isin(top_codes)]

        if len(top_geo) > 0:
            top_geo.plot(
                ax=ax,
                color='red',
                edgecolor='black',
                linewidth=2.0,
                alpha=0.7
            )

            # 행정동 라벨 추가
            for idx, row in top_geo.iterrows():
                try:
                    centroid = row.geometry.centroid
                    label = row.get('adm_nm', str(row.get(geo_key, 'Unknown')))
                    plt.annotate(
                        text=label,
                        xy=(centroid.x, centroid.y),
                        ha='center',
                        fontsize=16,
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="black", alpha=0.9)
                    )
                except Exception as e:
                    print(f"라벨 추가 중 오류: {e}")

        # ======= 추가 조치: 서울 좌표 경계에 맞게 영역 강제 설정 =======
        if len(seoul_only) > 0:
            # 서울만 포함한 데이터의 경계 계산
            bounds = seoul_only.total_bounds
            x_min, y_min, x_max, y_max = bounds

            # 약간의 여백 추가 (5%)
            x_padding = (x_max - x_min) * 0.05
            y_padding = (y_max - y_min) * 0.05

            # 좌표 영역 강제 설정
            ax.set_xlim([x_min - x_padding, x_max + x_padding])
            ax.set_ylim([y_min - y_padding, y_max + y_padding])

            print(f"서울 지역 좌표 범위: X({x_min} ~ {x_max}), Y({y_min} ~ {y_max})")
        else:
            # 좌표 데이터가 없으면 서울 대략적 좌표 사용 (하드코딩)
            print("서울 좌표 데이터 없음, 하드코딩된 서울 좌표 사용")
            ax.set_xlim([126.7648, 127.1839])  # 서울 경도 범위
            ax.set_ylim([37.4285, 37.7013])  # 서울 위도 범위

        # 그래프 제목 및 스타일
        plt.title('서울시 행정동별 그늘막 필요 지수', fontsize=24, fontweight='bold')
        plt.axis('off')  # 축 제거

        # 향상된 범례 추가
        from matplotlib.patches import Patch

        legend_elements = []

        cluster_descriptions = [
            "낮은 열 부하 지역 (클러스터 0)",
            "중하위 열 부하 지역 (클러스터 1)",
            "중간 열 부하 지역 (클러스터 2)",
            "중상위 열 부하 지역 (클러스터 3)",
            "높은 열 부하 지역 (클러스터 4)"
        ]

        for i in range(k):
            legend_elements.append(
                Patch(facecolor=cluster_colors[i], edgecolor='black',
                      label=cluster_descriptions[i])
            )

        legend_elements.append(
            Patch(facecolor='red', edgecolor='black',
                  label='그늘막 설치 최우선 행정동 (상위 3개)')
        )

        plt.legend(handles=legend_elements, loc='lower right', fontsize=14,
                   title="범례", title_fontsize=16)

        # 설명 텍스트 추가 - 개선된 설명
        description = (
            "색상 설명:\n"
            "- 각 색상은 열 환경 특성이 유사한 행정동 그룹(클러스터)을 나타냅니다\n"
            "- 클러스터 4(주황색)는 가장 더운 지역들로, 그늘막 필요성이 높은 지역입니다\n"
            "- 빨간색은 그늘막 필요 지수가 가장 높은 상위 3개 행정동입니다\n"
            "- 그늘막 필요 지수는 최대 온도(50%), 평균 온도(30%), 온도 변동 지수(20%)를 종합한 값입니다"
        )

        # 설명 텍스트 박스 추가
        plt.figtext(0.5, 0.01, description, ha='center', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 저장 옵션 개선
        plt.tight_layout()
        plt.savefig('seoul_shade_need_map.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("서울시 행정동별 그늘막 필요 지수 시각화 완료: seoul_shade_need_map.png 저장됨")

    except Exception as e:
        print(f"지도 시각화 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()  # 상세 오류 출력


    # 개별 행정동 지도 생성 코드
    def create_single_dong_map(dong_code, dong_name, data_copy, seoul_geo_copy, geo_key='adm_cd8', data_key='ADM_CD'):
        """특정 행정동만 강조하여 시각화하는 함수"""
        try:
            print(f"{dong_name}({dong_code}) 지도 생성 중...")

            # 대상 행정동 필터링
            target_dong = seoul_geo_copy[seoul_geo_copy[geo_key] == str(dong_code)]

            if len(target_dong) == 0:
                print(f"오류: {dong_code} 코드에 해당하는 행정동을 찾을 수 없습니다.")
                print(f"가능한 열: {seoul_geo_copy.columns.tolist()}")
                print(f"{geo_key} 열의 고유값 예시: {seoul_geo_copy[geo_key].unique()[:5]}")

                # 추가 디버깅: 행정동 코드 자리수 확인
                print(f"입력된 행정동 코드 길이: {len(str(dong_code))}")
                print(f"데이터의 행정동 코드 길이 예시: {seoul_geo_copy[geo_key].astype(str).str.len().value_counts()}")

                # 대안적 접근: 행정동 이름으로 시도
                if 'adm_nm' in seoul_geo_copy.columns and 'ADM_NM' in data_copy.columns:
                    target_dong = seoul_geo_copy[seoul_geo_copy['adm_nm'] == dong_name]
                    if len(target_dong) > 0:
                        print(f"행정동 이름으로 매칭 성공: {dong_name}")
                    else:
                        print(f"행정동 이름으로도 매칭 실패: {dong_name}")
                        return
                else:
                    return

            # 주변 행정동 포함 (버퍼 생성)
            buffer_size = 0.005  # 약 500m
            target_buffer = target_dong.geometry.buffer(buffer_size).unary_union

            # 버퍼와 교차하는 주변 행정동들 선택
            nearby_dongs = seoul_geo_copy[seoul_geo_copy.intersects(target_buffer)]

            # 시각화
            fig, ax = plt.subplots(figsize=(12, 10))

            # 주변 행정동 그리기
            nearby_dongs.plot(ax=ax, color='lightgrey', edgecolor='darkgrey', linewidth=1, alpha=0.5)

            # 대상 행정동 강조 (빨간색)
            target_dong.plot(ax=ax, color='red', edgecolor='black', linewidth=2)

            # 행정동 라벨 추가
            for idx, row in target_dong.iterrows():
                centroid = row.geometry.centroid
                plt.annotate(
                    text=dong_name,
                    xy=(centroid.x, centroid.y),
                    ha='center',
                    fontsize=18,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="black", alpha=0.9)
                )

                # 그늘막 필요 지수와 온도 변동 지수 값 표시 (수정됨)
                shade_idx = row.get('shade_need_index')
                temp_var = row.get('temp_variation')
                if shade_idx is not None and not pd.isna(shade_idx):
                    info_text = f'그늘막 필요 지수: {shade_idx:.2f}'
                    if temp_var is not None and not pd.isna(temp_var):
                        info_text += f'\n온도 변동 지수: {temp_var:.2f}°C'

                    plt.annotate(
                        text=info_text,
                        xy=(centroid.x, centroid.y - buffer_size / 2),  # 이름 아래에 표시
                        ha='center',
                        fontsize=14,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
                    )

            # 좌표 범위 설정
            bounds = nearby_dongs.total_bounds
            x_min, y_min, x_max, y_max = bounds

            # 약간의 여백 추가
            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.1

            # 좌표 영역 강제 설정
            ax.set_xlim([x_min - x_padding, x_max + x_padding])
            ax.set_ylim([y_min - y_padding, y_max + y_padding])

            # 그래프 제목 및 스타일
            plt.title(f'그늘막 설치 우선지역: {dong_name}', fontsize=20, fontweight='bold')
            plt.axis('off')  # 축 제거

            # 설명 추가
            plt.figtext(0.5, 0.01,
                        f"'{dong_name}'은(는) 열 부하가 가장 높은 지역 중 하나로, 그늘막 설치가 우선적으로 필요합니다.",
                        ha='center', fontsize=14,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # 저장 옵션 개선
            plt.tight_layout()
            file_name = f'priority_area_{dong_name.replace(" ", "_")}.png'
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"{dong_name} 지도 생성 완료: {file_name}")
            return file_name

        except Exception as e:
            print(f"{dong_name} 지도 생성 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None


    # 상위 3개 행정동에 대한 개별 지도 생성
    print("상위 3개 행정동 개별 지도 생성 시작...")
    for idx, row in top_3_dongs.iterrows():
        dong_code = row[data_key]
        dong_name = row['ADM_NM']
        create_single_dong_map(dong_code, dong_name, data_copy, seoul_geo_copy, geo_key, data_key)
else:
    print("행정동 경계 데이터가 없어 지도 시각화를 건너뜁니다.")
# 11. 종합 결과 저장
print("분석 결과와 시각화를 모두 저장했습니다.")
data.to_csv('Seoul_Temperature_Clustering_Results.csv', index=False, encoding='euc-kr')
top_3_dongs.to_csv('Top_3_Shade_Need_Dongs.csv', index=False, encoding='euc-kr')

# 결과 요약 출력
print("\n========== 분석 결과 요약 ==========")
print(f"전체 행정동 수: {len(data)}개")
print(f"클러스터 수: {k}개")
print(f"\n가장 더운 클러스터: 클러스터 {hottest_cluster}")
print(f"해당 클러스터 행정동 수: {len(data[data['cluster'] == hottest_cluster])}개")
print(f"해당 클러스터 평균 그늘막 필요 지수: {cluster_shade_need[hottest_cluster]:.2f}")

print("\n그늘막 설치 우선순위 상위 3개 행정동:")
for idx, row in top_3_dongs.iterrows():
    print(f"- {row['ADM_NM']} (행정동코드: {row['ADM_CD']})")
    print(f"  그늘막 필요 지수: {row['shade_need_index']:.2f}")
    print(f"  최대 온도: {row['max']:.2f}°C, 평균 온도: {row['mean']:.2f}°C")
    print(f"  온도 변동 지수: {row['temp_variation']:.2f}°C")  # 고온 영역 비율 대신 온도 변동 지수 표시

print("\n생성된 파일:")
print("- Seoul_Temperature_Clustering_Results.csv: 전체 클러스터링 결과")
print("- Top_3_Shade_Need_Dongs.csv: 상위 3개 행정동 상세 정보")
print("- elbow_method.png: 최적 클러스터 수 선정 그래프")
print("- cluster_heat_features.png: 클러스터별 열 특성 분포")
print("- seoul_temperature_tsne.png: t-SNE 기반 클러스터 시각화")  # 파일명 변경됨
print("- cluster_centers_heatmap.png: 클러스터 중심값 히트맵")
print("- cluster_shade_need_index.png: 클러스터별 그늘막 필요 지수")
print("- top_3_dongs_comparison.png: 상위 3개 행정동 특성 비교")

if has_geo_data:
    print("- seoul_shade_need_map.png: 서울시 행정동별 그늘막 필요 지수 지도")
    for idx, row in top_3_dongs.iterrows():
        dong_name = row['ADM_NM']
        print(f"- priority_area_{dong_name.replace(' ', '_')}.png: {dong_name} 상세 지도")

print("====================================")