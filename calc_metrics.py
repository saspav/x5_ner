import json
from seqeval.metrics import classification_report as seqeval_report
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer


def compute_metrics(data, predictions_output):
    true_labels = []
    true_predictions = []
    true_labels_B = []
    true_predictions_B = []

    mlb = MultiLabelBinarizer()

    for item, pred_output in zip(data, predictions_output):
        # Извлекаем оригинальные метки из data
        original_labels = [entity["entity_group"] for entity in item["entities"]]
        true_labels.append(original_labels)
        true_labels_B.append(["B-" + label for label in original_labels])

        # Извлекаем предсказанные метки из predictions_output
        predicted_labels = [entity["entity_group"] for entity in pred_output["entities"]]
        true_predictions.append(predicted_labels)
        true_predictions_B.append(["B-" + label for label in predicted_labels])

    # print(seqeval_report(true_labels_B, true_predictions_B))

    true_labels_bin = mlb.fit_transform(true_labels)
    true_predictions_bin = mlb.transform(true_predictions)

    report = classification_report(true_labels_bin, true_predictions_bin, zero_division=1)

    print(report)
    print(*enumerate(mlb.classes_))


if __name__ == "__main__":
    with open("data/test.json", encoding='utf-8') as file:
        test = json.load(file)

    with open("data/submission.json", encoding="utf-8") as outf:
        predict = json.load(outf)

    compute_metrics(test, predict)
