import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

np.random.seed(42)

token_length = np.random.randint(3, 101, size=150)

# Feature 2: Contains code snippets (1 = Yes, 0 = No)
contains_code_snippets = np.random.choice(
    [0, 1], size=150, p=[0.7, 0.3]
)

# Feature 3: Has a question mark (1 = Yes, 0 = No)
has_question_mark = np.random.choice(
    [0, 1], size=150, p=[0.2, 0.8]
)

target_labels = np.random.choice(
    [0,1,2], size=150, p=[0.3,0.3,0.4]
)

X = np.column_stack((
    token_length,
    contains_code_snippets,
    has_question_mark
))
y = target_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("---classification report---")
print(classification_report(y_test, y_pred, target_names=[
    "Simple General Chat",
    "Code Generation Request",
    "Complex Research Reasoning"
]))

sample_input = np.array([[150,1,0]])
prediction = model.predict(sample_input)

intent_categories = {
    0: "Simple General Chat",
    1: "Code Generation Request",
    2: "Complex Research Reasoning"
}
print(f"prediction for sameple input: {intent_categories[prediction[0]]}")