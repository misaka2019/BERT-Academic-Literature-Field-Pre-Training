from transformers import (BertForMaskedLM,
                          BertTokenizer,
                          TrainingArguments,
                          Trainer,
                          )
from util import TextDataset, DataCollator


def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_file)
    model = BertForMaskedLM.from_pretrained(args.bert_file)
    print('model of parameters: ', model.num_parameters())

    dataset = TextDataset(args.data_dir, tokenizer)
    data_collator = DataCollator(max_seq_len=args.max_seq_len, tokenizer=tokenizer, mlm_probability=0.15)
    print('data of lines: ', len(dataset))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        fp16_backend='auto',
        per_device_train_batch_size=args.per_device_train_batch_size,
        prediction_loss_only=True,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_dir=args.logging_dir,
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        # callbacks=[es],
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training arguments')

    parser.add_argument('--logging_dir', type=str, default='./logs', help='Directory to store logging information')
    parser.add_argument('--dataloader_num_workers', type=int, default=12,
                        help='Number of worker processes to use for loading data')
    parser.add_argument('--per_device_train_batch_size', type=int, default=128,
                        help='Batch size per GPU/CPU for training')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--save_total_limit', type=int, default=10, help='Total number of checkpoints to save')
    parser.add_argument('--bert_file', type=str, default='./chinese-roberta-wwm-ext',
                        help='Path to pre-trained BERT model file')
    parser.add_argument('--data_dir', type=str, default='./data/clean_text_list.txt',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./model', help='Directory to save trained model')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Maximum sequence length for input tokens')

    args = parser.parse_args()
    main(args)
