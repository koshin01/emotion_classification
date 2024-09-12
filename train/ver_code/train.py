from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import const


def load_model():
    emotions = const.EMOTIONS
    return AutoModelForSequenceClassification.from_pretrained(
        const.PRETRAINED_MODEL_NAME,
        num_labels=len(emotions),
    )


def exec(dataset):
    model = load_model()

    training_args = TrainingArguments(
        output_dir="./output/result",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()

    model.save_pretrained("./output/model")
