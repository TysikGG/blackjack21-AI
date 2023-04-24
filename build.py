import pandas as pd 
from io import StringIO

df = pd.read_csv('take.csv')
a = input('Введите значение первой карты в цифрах (если туз - 11) ')
b = input('Введите значение второй карты в цифрах (если туз - 11) ')
c = input('Введите значение карты диллера (если туз - 11) ')
test_df = pd.read_csv(StringIO('first_card,secound_card,diller_card,take\n' + a + ',' + b + ','+ c + ',' + '1'))

print(df.info())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

x = df.drop('result', axis = 1).drop('take_card', axis = 1)
y = df['result']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.01)
sc = StandardScaler()
print(test_df.info())
x_train = sc.fit_transform(x_train)
x_test = sc.transform(test_df)

classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(x_train, y_train)
print(x_test)
y_pred = classifier.predict(x_test)
print('\n')
if y_pred[0] == 0:
    print('\033[91m' + 'Не берите!' + '\033[0m')
else:
    print('\033[92m' + 'Берите!' + '\033[0m')
end = input('Введите любую клавишу для выхода . . .')
#percent = accuracy_score(y_test, y_pred) * 100
#print(percent)