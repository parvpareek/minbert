import random
import numpy as np
import argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask

TQDM_DISABLE = False
BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

class MultitaskBERT(nn.Module):
    '''
    Multitask BERT model for sentiment classification, paraphrase detection, and semantic textual similarity,
    enhanced with the SimCSE framework for improved sentence embeddings.
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Set BERT parameters' trainability based on config.option
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Assuming 5 sentiment classes; adjust if different
        self.sentiment_classifier = nn.Linear(config.hidden_size, 5)
        # Add heads for paraphrase and STS
        self.paraphrase_classifier = nn.Linear(3 * config.hidden_size, 1)
        self.sts_regressor = nn.Linear(3 * config.hidden_size, 1)
        

    def forward(self, input_ids, attention_mask):
        '''
        Forward pass to generate sentence embeddings.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            attention_mask (torch.Tensor): Attention mask for the input.

        Returns:
            torch.Tensor: Sentence embeddings from BERT's pooler output.
        '''
        output = self.bert(input_ids, attention_mask)
        embeddings = output.pooler_output  # [CLS] token embedding
        return embeddings

    def predict_sentiment(self, input_ids, attention_mask):
        '''
        Predict sentiment logits for a batch of sentences.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Logits for 5 sentiment classes.
        '''
        embeddings = self.forward(input_ids, attention_mask)
        embeddings = self.dropout(embeddings)
        logits = self.sentiment_classifier(embeddings)  # Shape: (batch_size, 5)
        return logits

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        '''
        Generate embeddings for paraphrase detection.

        Args:
            input_ids_1, input_ids_2 (torch.Tensor): Tokenized input IDs for sentence pairs.
            attention_mask_1, attention_mask_2 (torch.Tensor): Attention masks.

        Returns:
            tuple: (emb1, emb2) embeddings for the two sentences.
        '''
        emb1 = self.forward(input_ids_1, attention_mask_1)  # Shape: (batch_size, hidden_size)
        emb2 = self.forward(input_ids_2, attention_mask_2)  # Shape: (batch_size, hidden_size)
        abs_diff = torch.abs(emb1 - emb2)  # Element-wise absolute difference
        combined = torch.cat([emb1, emb2, abs_diff], dim=1)  # Shape: (batch_size, 3 * hidden_size)
        combined = self.dropout(combined)
        logits = self.paraphrase_classifier(combined)  # Shape: (batch_size, 1)
        return logits

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        '''
        Generate embeddings for semantic textual similarity.

        Args:
            input_ids_1, input_ids_2 (torch.Tensor): Tokenized input IDs for sentence pairs.
            attention_mask_1, attention_mask_2 (torch.Tensor): Attention masks.

        Returns:
            tuple: (emb1, emb2) embeddings for the two sentences.
        '''
        emb1 = self.forward(input_ids_1, attention_mask_1)  # Shape: (batch_size, hidden_size)
        emb2 = self.forward(input_ids_2, attention_mask_2)  # Shape: (batch_size, hidden_size)
        abs_diff = torch.abs(emb1 - emb2)  # Element-wise absolute difference
        combined = torch.cat([emb1, emb2, abs_diff], dim=1)  # Shape: (batch_size, 3 * hidden_size)
        combined = self.dropout(combined)
        scores = self.sts_regressor(combined)  # Shape: (batch_size, 1)
        return scores

def compute_simcse_loss(model, input_ids, attention_mask, device, tau=0.05):
    '''
    Compute the SimCSE contrastive loss for a batch of sentences.

    Args:
        model (MultitaskBERT): The model instance.
        input_ids (torch.Tensor): Tokenized input IDs of shape (B, seq_len).
        attention_mask (torch.Tensor): Attention mask of shape (B, seq_len).
        device (torch.device): Device to run computations on.
        tau (float): Temperature parameter for scaling similarity.

    Returns:
        torch.Tensor: Contrastive loss value.
    '''
    B = input_ids.size(0)
    # Duplicate the batch to create 2B samples
    input_ids_2b = torch.cat([input_ids, input_ids], dim=0).to(device)
    attention_mask_2b = torch.cat([attention_mask, attention_mask], dim=0).to(device)
    # Get embeddings for the duplicated batch
    embeddings_2b = model.forward(input_ids_2b, attention_mask_2b)  # Shape: (2B, hidden_size)
    # Split into two sets of embeddings
    z1 = embeddings_2b[:B]  # First pass
    z2 = embeddings_2b[B:]  # Second pass with different dropout
    # Normalize embeddings to unit vectors for cosine similarity
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    # Compute similarity matrix between z1 and z2
    sim_matrix = torch.mm(z1, z2.t()) / tau  # Shape: (B, B)
    # Labels: each z1[i] should be most similar to z2[i]
    labels = torch.arange(B).to(device)
    # Cross-entropy loss over similarity scores
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")
def train_multitask(args):
    # Set device
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Load multitask data
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split='train'
    )
    sst_dev_data, _, para_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split='dev'
    )

    # Create datasets
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    # Create data loaders
    sst_train_dataloader = DataLoader(
        sst_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_data.collate_fn
    )
    sst_dev_dataloader = DataLoader(
        sst_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_data.collate_fn
    )
    para_train_dataloader = DataLoader(
        para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn
    )
    para_dev_dataloader = DataLoader(
        para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn
    )
    sts_train_dataloader = DataLoader(
        sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn
    )
    sts_dev_dataloader = DataLoader(
        sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn
    )

    # Initialize model configuration
    config = {
        'hidden_dropout_prob': args.hidden_dropout_prob,
        'num_labels': num_labels,
        'hidden_size': 768,
        'data_dir': '.',
        'option': args.option
    }
    config = SimpleNamespace(**config)
    model = MultitaskBERT(config)
    model = model.to(device)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_dev_acc = 0
    lambda_simcse = 1.0  # Weight for SimCSE loss

    # Define loss functions
    sentiment_loss_fn = nn.CrossEntropyLoss()  # For sentiment classification (multi-class)
    paraphrase_loss_fn = nn.BCEWithLogitsLoss()  # For paraphrase detection (binary classification)
    sts_loss_fn = nn.MSELoss()  # For STS (regression)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        # Iterate over batches from all tasks
        for sst_batch, para_batch, sts_batch in tqdm(
            zip(sst_train_dataloader, para_train_dataloader, sts_train_dataloader),
            desc=f'train-{epoch}',
            disable=TQDM_DISABLE):
            ### Sentiment Classification (SST)
            sst_ids = sst_batch['token_ids'].to(device)
            sst_mask = sst_batch['attention_mask'].to(device)
            sst_labels = sst_batch['labels'].to(device)  # Shape: (batch_size,)
            sentiment_logits = model.predict_sentiment(sst_ids, sst_mask)  # Shape: (batch_size, 5)
            sent_loss = sentiment_loss_fn(sentiment_logits, sst_labels)
            # SimCSE loss for SST
            simcse_loss_sst = compute_simcse_loss(model, sst_ids, sst_mask, device)

            ### Paraphrase Detection (Quora)
            para_ids1 = para_batch['token_ids_1'].to(device)
            para_mask1 = para_batch['attention_mask_1'].to(device)
            para_ids2 = para_batch['token_ids_2'].to(device)
            para_mask2 = para_batch['attention_mask_2'].to(device)
            para_labels = para_batch['labels'].to(device).float()  # Shape: (batch_size,), 0 or 1
            paraphrase_logits = model.predict_paraphrase(para_ids1, para_mask1, para_ids2, para_mask2)  # Shape: (batch_size, 1)
            paraphrase_logits = paraphrase_logits.squeeze(1)  # Shape: (batch_size,)
            para_loss = paraphrase_loss_fn(paraphrase_logits, para_labels)
            # SimCSE loss for Quora
            simcse_loss_para1 = compute_simcse_loss(model, para_ids1, para_mask1, device)
            simcse_loss_para2 = compute_simcse_loss(model, para_ids2, para_mask2, device)
            simcse_loss_para = (simcse_loss_para1 + simcse_loss_para2) / 2

            ### Semantic Textual Similarity (STS)
            sts_ids1 = sts_batch['token_ids_1'].to(device)
            sts_mask1 = sts_batch['attention_mask_1'].to(device)
            sts_ids2 = sts_batch['token_ids_2'].to(device)
            sts_mask2 = sts_batch['attention_mask_2'].to(device)
            sts_labels = sts_batch['labels'].to(device).float()  # Shape: (batch_size,), e.g., 0 to 5
            similarity_scores = model.predict_similarity(sts_ids1, sts_mask1, sts_ids2, sts_mask2)  # Shape: (batch_size, 1)
            similarity_scores = similarity_scores.squeeze(1)  # Shape: (batch_size,)
            sts_loss = sts_loss_fn(similarity_scores, sts_labels)
            # SimCSE loss for STS
            simcse_loss_sts1 = compute_simcse_loss(model, sts_ids1, sts_mask1, device)
            simcse_loss_sts2 = compute_simcse_loss(model, sts_ids2, sts_mask2, device)
            simcse_loss_sts = (simcse_loss_sts1 + simcse_loss_sts2) / 2

            ### Combine Losses
            total_simcse_loss = simcse_loss_sst + simcse_loss_para + simcse_loss_sts
            total_loss = sent_loss + para_loss + sts_loss + lambda_simcse * total_simcse_loss

            ### Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            num_batches += 1

        # Compute average training loss
        train_loss = train_loss / num_batches

        # Evaluate on development set
        dev_metrics = model_eval_multitask(
            sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device
        )

        # Save model if SST accuracy improves
        if dev_metrics['sst_acc'] > best_dev_acc:
            best_dev_acc = dev_metrics['sst_acc']
            save_model(model, optimizer, args, config, args.filepath)

        # Print epoch results
        print(
            f"Epoch {epoch}: train loss :: {train_loss:.3f}, "
            f"dev sst acc :: {dev_metrics['sst_acc']:.3f}, "
            f"dev para acc :: {dev_metrics['para_acc']:.3f}, "
            f"dev sts corr :: {dev_metrics['sts_corr']:.3f}"
        )
        
        
def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels, para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test, args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels, para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy, dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: BERT parameters frozen; finetune: BERT parameters updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt'
    seed_everything(args.seed)
    train_multitask(args)
    test_multitask(args)