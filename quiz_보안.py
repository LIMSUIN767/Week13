# 퀴즈를 시간 내에 다 작성하지 못했습니다.  
# 며칠간 감기로 인해 컨디션이 좋지 않아 충분한 공부를 하지 못한 점 양해 부탁드립니다.  
# 그래도 늦었지만 끝까지 최선을 다해 풀고자 하는 마음으로 제출합니다. 감사합니다.

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

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'AppleGothic'  
plt.rcParams['axes.unicode_minus'] = False

import seaborn as sns
import matplotlib.pyplot as plt
import os
sns.boxplot(data=df_selected[['이직여부','성별','야근여부','집까지거리','월급여','워라밸','근무환경만족도']])
plt.savefig("boxplot_features.png")
os.system("open boxplot_features.png")

# 모델
# 1단계: DataFrame을 NumPy 배열로 변환하기
raw = df_selected
np_raw = raw.values
type(np_raw)

# 데이터 표준화
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# standardScaler
X = df_selected.drop('이직여부', axis=1)
ss = StandardScaler()
X_ss = ss.fit_transform(X)
X_ss_pd = pd.DataFrame(X_ss, columns=X.columns)

## 학습 데이터(X)와 정답 데이터(y) 분리하기
y=raw['이직여부']
X=raw.drop(['이직여부'], axis=1)
X.head()

y = df_selected['이직여부']
X = X_ss_pd  # ✅ 이걸 써야 정규화가 반영됨

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
#standardscaler
X_out=X_ss_pd
X_train, X_test, y_train, y_test=\
train_test_split(X_out, y, test_size=0.2, random_state=13)
log_reg=LogisticRegression(random_state=13, solver='liblinear', C=10.)
log_reg.fit(X_train, y_train)
pred=log_reg.predict(X_test)
accuracy_score(y_test, pred)
print(accuracy_score(y_test, pred))

#6. 성능 검사
# 1. 로지스틱 회귀: 표준화하지 않은 데이터 사용
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=13, solver='liblinear', C=10.0)
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
# 이직여부=Yes로 예측된 직원 수를 출력하시오. -> 166
from sklearn.linear_model import LogisticRegression

# 1. 모델 정의 및 학습
import numpy as np
log_reg = LogisticRegression(solver='liblinear', random_state=13, C=10.0)
log_reg.fit(X_train, y_train)

# 5명
proba = log_reg.predict_proba(X_test)[:, 1] 
top_5_idx = np.argsort(proba)[::-1][:5]
top_5_df = X_test.iloc[top_5_idx].copy()
top_5_df["이직여부"] = proba[top_5_idx]
print(top_5_df)

# 7.
feature_names = ['이직여부','성별','야근여부','집까지거리','월급여','워라밸','근무환경만족도']
data = [
    {
        "Age": 29, "BusinessTravel": "Travel_Rarely", "Department": "Research & Development",
        "DistanceFromHome": 5, "Education": 3, "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 2, "Gender": "Male", "HourlyRate": 70,
        "JobInvolvement": 3, "JobLevel": 1, "JobRole": "Laboratory Technician",
        "JobSatisfaction": 2, "MaritalStatus": "Single", "MonthlyIncome": 2800,
        "NumCompaniesWorked": 1, "OverTime": "Yes", "PercentSalaryHike": 12,
        "PerformanceRating": 3, "RelationshipSatisfaction": 2, "StockOptionLevel": 0,
        "TotalWorkingYears": 4, "TrainingTimesLastYear": 2, "WorkLifeBalance": 2,
        "YearsAtCompany": 1, "YearsInCurrentRole": 1, "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 1
    },
    {
        "Age": 42, "BusinessTravel": "Non-Travel", "Department": "Human Resources",
        "DistanceFromHome": 10, "Education": 4, "EducationField": "Human Resources",
        "EnvironmentSatisfaction": 3, "Gender": "Female", "HourlyRate": 85,
        "JobInvolvement": 3, "JobLevel": 3, "JobRole": "Human Resources",
        "JobSatisfaction": 4, "MaritalStatus": "Married", "MonthlyIncome": 5200,
        "NumCompaniesWorked": 2, "OverTime": "No", "PercentSalaryHike": 14,
        "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 1,
        "TotalWorkingYears": 18, "TrainingTimesLastYear": 3, "WorkLifeBalance": 3,
        "YearsAtCompany": 7, "YearsInCurrentRole": 4, "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3
    },
    {
        "Age": 35, "BusinessTravel": "Travel_Frequently", "Department": "Sales",
        "DistanceFromHome": 2, "Education": 2, "EducationField": "Marketing",
        "EnvironmentSatisfaction": 1, "Gender": "Male", "HourlyRate": 65,
        "JobInvolvement": 2, "JobLevel": 2, "JobRole": "Sales Executive",
        "JobSatisfaction": 1, "MaritalStatus": "Single", "MonthlyIncome": 3300,
        "NumCompaniesWorked": 3, "OverTime": "Yes", "PercentSalaryHike": 11,
        "PerformanceRating": 3, "RelationshipSatisfaction": 2, "StockOptionLevel": 0,
        "TotalWorkingYears": 10, "TrainingTimesLastYear": 2, "WorkLifeBalance": 2,
        "YearsAtCompany": 2, "YearsInCurrentRole": 1, "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 1
    }
] 
# ------------------------------------------------------ 여기서부터 시작 ---------------------------

df_Data = pd.DataFrame(data)
print(df_Data)

df_sample = df_Data[['Gender','OverTime','DistanceFromHome','MonthlyIncome','WorkLifeBalance','EnvironmentSatisfaction']]
df_sample.rename(columns={
    'Gender': '성별',
    'OverTime': '야근여부',
    'DistanceFromHome': '집까지거리',
    'MonthlyIncome': '월급여',
    'WorkLifeBalance': '워라밸',
    'EnvironmentSatisfaction': '근무환경만족도'
}, inplace=True)
df_sample = df_sample[['성별','야근여부','집까지거리','월급여','워라밸','근무환경만족도']]

print(df_sample)

df_sample['성별'] = df_sample['성별'].map({'Male': 1, 'Female': 0})
df_sample['야근여부'] = df_sample['야근여부'].map({'Yes': 1, 'No': 0})

df_scaled = ss.transform(df_sample)

# 예측
pred = log_reg.predict(df_scaled)
proba = log_reg.predict_proba(df_scaled)

df_sample['이직예측(0:유지,1:이직)'] = pred
df_sample['이직확률'] = proba[:, 1]
print(df_sample)


# 9번
coefficients = log_reg.coef_[0]
features = X_train.columns
importance = pd.Series(coefficients, index=features).sort_values(key=abs, ascending=False)
print(importance)

# 야근여부가 제일 큰 영향을 끼치는 것으로 나옵니다. 야근을 할수록 근무하고 싶어지는 마음이 사라지는 것으로 생각이 되어 이직할 확률을 높이는 것 같습니다.
# 두번째로 집까지거리 입니다. 집에서 멀면 멀수록 출근하는 시간이 길어집니다. 그 시간을 이동시간을 사용한다는 것이 이직확률을 높이지 않았나 싶습니다.
# 세번째로 성별입니다. 성별의 여부가 큰 영향은 주지 않지만 그래도 3번째로 영향을 준다고 나옵니다.