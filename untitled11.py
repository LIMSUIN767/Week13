# 1단계: 데이터 불러오기 및 컬럼 정리
import pandas as pd
url='https://raw.githubusercontent.com/sehakflower/data/main/titanic.csv'
titanic_df = pd.read_csv(url, sep='\t')  # 탭으로 구분된 CSV 파일 읽기

print(titanic_df)

# 1단계: 데이터 불러오기 및 컬럼 정리
# 컬럼명을 소문자로 통일
new_columns = ['passengerId', 'survived', 'pclass', 'name', 'sex', 'age',
               'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked']
titanic_df.columns = new_columns

# 2단계: 필요한 열만 추출
titanic_df1 = titanic_df[['survived', 'pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'fare']]

# 3단계: 성별을 숫자형(gender)으로 변환
# 성별을 숫자로! 머신러닝에서는 숫자로만 가능!
tmp = []
for each in titanic_df1['sex']:
    tmp.append(0 if each == 'female' else 1)

titanic_df1['gender'] = tmp
titanic_df1.drop(columns='sex', inplace=True)

titanic_df1['name'].unique()

# 4단계: 이름에서 호칭(title) 추출 및 변환
condition = lambda x: x.split(',')[1].split('.')[0].strip()
titanic_df1['title'] = titanic_df1['name'].map(condition)

# 일부 특수 호칭은 'Special'로 통합
Special = ['Master', 'Don', 'Rev']
for each in Special:
    titanic_df1['title'] = titanic_df1['title'].replace(each, 'Special')

titanic_df1.drop('name', axis=1, inplace=True)

titanic_df1['title'].unique()

# 5단계: 호칭을 숫자형으로 변환
def convert_title(x):
    return 1 if x == "Special" else 0

titanic_df1['special_title'] = titanic_df1['title'].apply(convert_title)
titanic_df1.drop('title', axis=1, inplace=True)

print(titanic_df1)

#6단계: 동반자 수(sibsp + parch)를 하나의 열로 통합
# 승객이 혼자 탔는지 아니면 가족이랑 탔는지..
titanic_df1['sibpar'] = titanic_df1['sibsp'] + titanic_df1['parch']
titanic_df1.drop(['sibsp', 'parch'], axis=1, inplace=True)

# 7단계: 평균 탑승 요금 계산
titanic_df1['n_family'] = titanic_df1['sibpar'] + 1  # 자신 포함
titanic_df1['avgfare'] = titanic_df1['fare'] / titanic_df1['n_family']
titanic_df1.drop(['fare', 'sibpar'], axis=1, inplace=True)

# 8단계: 컬럼명 및 순서 정리
# 컬럼명 바꾸는 것!
titanic_df1.rename(columns={
    'gender': 'sex',
    'special_title': 'title',
    'avgfare': 'fare',
    'n_family': 'num_family'
}, inplace=True)
#  순서 바꾸기
titanic_df1 = titanic_df1[['survived', 'pclass', 'sex', 'age', 'title', 'fare', 'num_family']]

# 9단계: 결측치(NaN) 제거
titanic_df1 = titanic_df1.dropna()

titanic_df1.head()

"""**Pandas → NumPy 변환, 그리고 직접 학습용/테스트용 데이터 분할**"""

# 1단계: DataFrame을 NumPy 배열로 변환하기
raw = titanic_df1
np_raw = raw.values
type(np_raw)

#  2단계: 학습 데이터와 테스트 데이터 나누기
train = np_raw[:100]
test = np_raw[100:]

# 3단계: 입력값(X)과 정답값(y) 분리
y_train = [i[0] for i in train]   # 생존 여부 (정답) → 첫 번째 컬럼
X_train = [j[1:] for j in train]  # 나머지 특성들 (입력값)

y_test = [i[0] for i in test]     # 테스트용 정답
X_test = [j[1:] for j in test]    # 테스트용 입력값

# 4단계: 데이터 크기 확인

len(X_train), len(y_train), len(y_test), len(X_test)

"""**의사결정 나무**"""


# 1단계: 의사결정나무 모델 설치 및 학습
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
model.fit(X_train, y_train)

print('Score:', model.score(X_train, y_train))
print('Score:', model.score(X_test, y_test))


from sklearn.tree import export_graphviz
import graphviz

export_graphviz(
    model,
    out_file="titanic.dot",
    feature_names=['pclass', 'sex', 'age', 'title', 'fare', 'num_family'],
    class_names=['0', '1'],
    rounded=True,
    filled=True
)