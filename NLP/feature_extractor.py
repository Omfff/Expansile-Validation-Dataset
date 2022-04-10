from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader
from train_eval import train
from dataset_pool import get_dataset
from utils import PathConfig

FE_SAVE_PATH = PathConfig().get_reuters_fe_path()


def load_feature_extractor(weight_path=FE_SAVE_PATH):
    model = AutoModelForSequenceClassification.from_pretrained(weight_path, output_hidden_states=True)
    return model


def train_feature_extractor(device, model_name, dataset_name, num_classes, model_save_path):
    train_dst, _ = get_dataset(dataset_name, model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    model.to(device)
    train_dst.set_format("torch")
    train_dataloader = DataLoader(train_dst, shuffle=True, batch_size=8)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    num_epochs = 10
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(train_dataloader)
    )
    _, _, _, model = train(model, train_dataloader, None, None, optimizer, lr_scheduler, num_epochs, device)
    model.save_pretrained(save_directory=model_save_path)


def main():
    device = "cuda:0"
    model_name = "bert-base-uncased"
    dataset_name = 'wheat_corn_reuters'
    num_classes = 2
    train_feature_extractor(device, model_name, dataset_name, num_classes,
                            model_save_path=FE_SAVE_PATH)


if __name__ == '__main__':
    main()