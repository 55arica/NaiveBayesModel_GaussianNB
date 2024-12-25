import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('weatherHistory.csv')

encoder = LabelEncoder()

df['Formatted Date'] = encoder.fit_transform(df['Formatted Date'])
df['Summary'] = encoder.fit_transform(df['Summary'])
df['Precip Type'] = encoder.fit_transform(df['Precip Type'])
df['Daily Summary'] = encoder.fit_transform(df['Daily Summary'])


x = df.drop(columns=['Daily Summary'])
y = df['Daily Summary']

x_train, x_test, y_train, y_test  =train_test_split(x, y, test_size = 0.2, random_state = 42)

model = GaussianNB()

model.fit(x_train, y_train)

predictions = model.predict(x_test)


accuracy_score = accuracy_score(y_test, predictions)

classification_report = classification_report(y_test, predictions)

print(f'Classification Report : {classification_report}')
print(f'Accuracy Score : {accuracy_score}')
