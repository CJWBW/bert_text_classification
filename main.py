import torch
import pickle
from transformers import BertTokenizer
from sklearn.metrics import classification_report
from data_processor.data_processor import DataProcessor
from bert.model import BERTClassifier, Model


NUM_LABELS = 2
BATCH_SIZE = 8
EPOCHS = 4
MAX_LEN = 64


def get_device():
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
        return device

    else:
        # print('No GPU available, using the CPU instead.')
        # device = torch.device("cpu")
        print('no GPU')
        exit()


def main():

    class_names = ['True', 'Fake']
    data_processor = DataProcessor()
    labels, statements, metadata, states, affiliations, credit_count = data_processor.load_dataset()
    # convert text labels to 0-5
    labels = {'train': DataProcessor.convert_labels(NUM_LABELS, labels['train']),
              'test': DataProcessor.convert_labels(NUM_LABELS, labels['test']),
              'validation': DataProcessor.convert_labels(NUM_LABELS, labels['validation'])}

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    train_dataloader = DataProcessor.create_dataloader(statements['train'], labels['train'], metadata['train'], states['train'], affiliations['train'], credit_count['train'], tokenizer, MAX_LEN, BATCH_SIZE)

    test_dataloader = DataProcessor.create_dataloader(statements['test'], labels['test'], metadata['test'], states['test'], affiliations['test'], credit_count['test'], tokenizer, MAX_LEN, BATCH_SIZE)

    validation_dataloader = DataProcessor.create_dataloader(statements['validation'], labels['validation'], metadata['validation'], states['validation'], affiliations['validation'], credit_count['validation'], tokenizer,
                                                            MAX_LEN, BATCH_SIZE)

    device = get_device()
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Train model

    model = BERTClassifier(model_option="bert-base-uncased", n_classes=NUM_LABELS)

    model = model.to(device)
    train_history = Model.train_model(model, train_dataloader, validation_dataloader, len(statements['train']), len(statements['validation']), EPOCHS, device, loss_fn)

    # evaluate model on test dataset
    test_acc, _ = Model.eval_model(model, test_dataloader, len(statements['test']), device, loss_fn)
    print('test accuracy: ', test_acc.item())

    # predictions
    pred, test_labels = Model.get_predictions(model, test_dataloader, device)

    print(classification_report(test_labels, pred, target_names=class_names))
    with open('recordnew.txt', 'wb') as f:
        pickle.dump(pred, f)
        pickle.dump(test_labels, f)


if __name__ == "__main__":
    main()
