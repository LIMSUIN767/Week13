# 1단계: 데이터 불러오기 및 컬럼 정리
import pandas as pd
df = pd.read_csv('/Users/imsu-in/Downloads/myproject/Week13-1/week13.csv')
print(df) # 링크에 오류가 생겨 다운받아서 로드했습니다.

# 2단계 전처리
print(df.isnull().sum())
df.dropna(inplace=True)

# 이직여부
print(df['이직여부'].unique())
tmp = []
for each in df['이직여부']:
    tmp.append(1 if each == 'Yes' else 0)
df['이직여부_숫자'] = tmp # 세로운 컬럼
df.drop(columns='이직여부', inplace=True)
df.rename(columns={'이직여부_숫자':'이직여부'}, inplace=True)
print(df['이직여부'])
# 나머지 데이터 전처리는 feature를 고르면서 할 예정

# 3단계 피처선택
# 성별  # 큰 영향은 없겠지만 그래도 야근과의 상호작용이 있을거라고 판단
tmp_sex = []
for each in df['성별']:
    tmp_sex.append(0 if each == 'Female' else 1)

df['성별_숫자'] = tmp_sex # 세로운 컬럼
df.drop(columns='성별', inplace=True)
df.rename(columns={'성별_숫자':'성별'}, inplace=True)
print(df['성별'])

# 야근여부   # 큰 영향이 있을거라고 생각
tmp_time = []
for each in df['야근여부']:
    tmp_time.append(0 if each == 'No' else 1)

df['야근여부_숫자'] = tmp_time # 세로운 컬럼
df.drop(columns='야근여부', inplace=True)
df.rename(columns={'야근여부_숫자':'야근여부'}, inplace=True)
print(df['야근여부'])

# 집까지거리(수치형)   # 멀면 이직률이 높아질거라고 생각
# 월급여(수치형)    # 높으면 이직률이 낮아질거라고 생각
# 워라밸(숫자형)   #높으면 이직률이 낮아질거라고 생각
# 근무환경만족도(숫자형)   #높으면 이직률이 낮아질거라고 생각

df_selected = df[['이직여부','성별','야근여부','집까지거리','월급여','워라밸','근무환경만족도']]
print(df_selected)
df_selected.dropna()


# 데이터 분포 확인


# 모델
# 1단계: DataFrame을 NumPy 배열로 변환하기
raw = df_selected
np_raw = raw.values
type(np_raw)

## 학습 데이터(X)와 정답 데이터(y) 분리하기
y=raw['이직여부']
X=raw.drop(['이직여부'], axis=1)
X.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)