import numpy as np
import torch
import torch.nn.functional as F
import os

import argparse
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import wandb
import datetime
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from huggingface_hub import hf_hub_download
import zipfile

from data import load_multiple
from utils import compute_metrics_np
from contrastive import ContrastiveModule

import random, os
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, patience=7,delta=0, verbose=True , path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation accuracy improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = float('-inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score >= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        """Saves model when validation accuracy increase."""
        if self.verbose:
            self.trace_func(f'Validation accuracy increased ({self.val_acc_max:.6f}% --> {val_acc:.6f}%). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_acc_max = val_acc

def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def check_path(paths):
    if not os.path.exists(paths):
        # Create the directory
        os.makedirs(paths)
        print(f'Directory {paths} created')
    else:
        print(f'Directory {paths} already exists')

def main(args):
    save_path_alvis = os.getenv('UNIMTS_DATA_ROOT') + f'models/UNIMTS/{args.dataset}/LOSOEvaluation/{args.experiment}/{args.seed}/'
    path_alvis = os.path.join(save_path_alvis, f'model_classifier_{args.SensorPosition}.pth')
    save_path = path_alvis
    early_stopping = EarlyStopping(patience=5, delta=0.001, path=path_alvis)
    repo_id = "xiyuanz/UniMTS"
    checkpoint_file = "checkpoint/UniMTS.pth"
    config_file = "config.json"
    data_file = "UniMTS_data.zip"

    

    check_path(save_path_alvis)

    if not os.path.exists("checkpoint"):
        hf_hub_download(repo_id=repo_id, filename=checkpoint_file, local_dir="./")
    hf_hub_download(repo_id=repo_id, filename=config_file, local_dir="./")
    if not os.path.exists("UniMTS_data"):
        hf_hub_download(repo_id=repo_id, filename=data_file, local_dir="./")
        with zipfile.ZipFile("UniMTS_data.zip", 'r') as zip_ref:
            zip_ref.extractall("./")

    # load real data
    # dataset_list = ['Opp_g','UCIHAR','MotionSense','w-HAR','Shoaib','har70plus','realworld','TNDA-HAR','PAMAP',\
    #                 'USCHAD','Mhealth','Harth','ut-complex','Wharf','WISDM','DSADS','UTD-MHAD','MMAct']
    dataset_list = [args.dataset]
    train_inputs_list, train_masks_list, train_labels_list, label_list_list, all_text_list, num_classes_list = load_multiple(dataset_list, args.padding_size, args.data_path, args.experiment, split='train', k=args.k, args= args)
    validation_inputs_list, validation_masks_list, validation_labels_list, label_list_list, all_text_list, num_classes_list = load_multiple(dataset_list, args.padding_size, args.data_path,args.experiment, split='validation', k=args.k, args= args)
    test_inputs_list, test_masks_list, test_labels_list, label_list_list, all_text_list, _ = load_multiple(dataset_list, args.padding_size, args.data_path,args.experiment, split='test', args = args)
    train_dataloader_list, validation_dataloader_list, test_dataloader_list = [], [], []
    for real_inputs, real_masks, real_labels in zip(train_inputs_list, train_masks_list, train_labels_list):
        train_dataset = TensorDataset(real_inputs, real_masks, real_labels)
        train_dataloader_list.append(DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True))
    for real_inputs, real_masks, real_labels in zip(validation_inputs_list, validation_masks_list, validation_labels_list):
        validation_dataset = TensorDataset(real_inputs, real_masks, real_labels)
        validation_dataloader_list.append(DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False))
    for real_inputs, real_masks, real_labels in zip(test_inputs_list, test_masks_list, test_labels_list):
        test_dataset = TensorDataset(real_inputs, real_masks, real_labels)
        test_dataloader_list.append(DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False))

    date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
    run_name = f"{save_path_alvis}/{args.stage}_{args.mode}" + f"{date}"
    print("W&B Run Name:", run_name)

    wandb.init(
        project='UniMTS',
        name=run_name,
        mode='offline' 
    )

    print("save_path_alvis:", save_path_alvis)
    print("run_tag:", args.run_tag)
    print("stage:", args.stage)
    print("mode:", args.mode)
    print("k:", args.k)
    print("Date:", date)


    save_path = './checkpoint/%s/' % args.run_tag
    os.makedirs(save_path, exist_ok=True)

    for ds, train_dataloader, validation_dataloader, test_dataloader, validation_labels, test_labels, label_list, all_text, num_class in \
            zip(dataset_list, train_dataloader_list, validation_dataloader_list, test_dataloader_list, validation_labels_list, test_labels_list, label_list_list, all_text_list, num_classes_list):
        
        args.num_class = num_class
        model = ContrastiveModule(args).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        if args.mode == 'full' or args.mode == 'probe':
            model.model.load_state_dict(torch.load(f'{args.checkpoint}'))
        if args.mode == 'probe':
            for name, param in model.model.named_parameters():
                param.requires_grad = False

        best_loss = None
        for epoch in range(args.num_epochs):

            tol_loss = 0
            
            model.train()
            for i, (input, mask, label) in enumerate(train_dataloader):

                input = input.to(args.device)
                labels = label.to(args.device)

                if not args.gyro:
                    b, t, c = input.shape
                    indices = np.array([range(i, i+3) for i in range(0, c, 6)]).flatten()
                    input = input[:,:,indices]

                b, t, c = input.shape
                if args.stft:
                    input_stft = input.permute(0,2,1).reshape(b * c,t)
                    input_stft = torch.abs(torch.stft(input_stft, n_fft = 25, hop_length = 28, onesided = False, center = True, return_complex = True))
                    input_stft = input_stft.reshape(b, c, input_stft.shape[-2], input_stft.shape[-1]).reshape(b, c, t).permute(0,2,1)
                    input = torch.cat((input, input_stft), dim=-1)

                input = input.reshape(b, t, 22, -1).permute(0, 3, 1, 2).unsqueeze(-1)
              
                output = model.classifier(input)
             
                loss = F.cross_entropy(output.float(), labels.long(), reduction="mean")
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tol_loss += len(input) * loss.item()
            
                # print(epoch, i, loss.item())
            
            print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {tol_loss / len(train_dataset):.4f}')
            wandb.log({'{ds} loss': tol_loss / len(train_dataset)})

            # if best_loss is None or tol_loss < best_loss:
            #     best_loss = tol_loss
            #     torch.save(model.state_dict(), os.path.join(save_path_alvis, f'model_classifier_{args.SensorPosition}.pth'))


            model.eval()
            with torch.no_grad():

                pred_whole, logits_whole = [], []
                for input, mask, label in validation_dataloader:
                    
                    input = input.to(args.device)
                    label = label.to(args.device)

                    if not args.gyro:
                        b, t, c = input.shape
                        indices = np.array([range(i, i+3) for i in range(0, c, 6)]).flatten()
                        input = input[:,:,indices]

                    b, t, c = input.shape
                    if args.stft:
                        input_stft = input.permute(0,2,1).reshape(b * c,t)
                        input_stft = torch.abs(torch.stft(input_stft, n_fft = 25, hop_length = 28, onesided = False, center = True, return_complex = True))
                        input_stft = input_stft.reshape(b, c, input_stft.shape[-2], input_stft.shape[-1]).reshape(b, c, t).permute(0,2,1)
                        input = torch.cat((input, input_stft), dim=-1)

                    input = input.reshape(b, t, 22, -1).permute(0, 3, 1, 2).unsqueeze(-1)

                    logits_per_imu = model.classifier(input)
                    logits_whole.append(logits_per_imu)
                    loss = F.cross_entropy(logits_per_imu.float(), label.long(), reduction="mean")
                    tol_loss += len(input) * loss.item()
                    
                    pred = torch.argmax(logits_per_imu, dim=-1).detach().cpu().numpy()
                    pred_whole.append(pred)

                pred = np.concatenate(pred_whole)
                acc = accuracy_score(validation_labels, pred)
                prec = precision_score(validation_labels, pred, average='macro')
                rec = recall_score(validation_labels, pred, average='macro')
                f1 = f1_score(validation_labels, pred, average='weighted')

                print(f"{ds} acc: {acc}, {ds} prec: {prec}, {ds} rec: {rec}, {ds} f1: {f1}")
                wandb.log({f"{ds} acc": acc, f"{ds} prec": prec, f"{ds} rec": rec, f"{ds} f1": f1})

                logits_whole = torch.cat(logits_whole)
                r_at_1, r_at_2, r_at_3, r_at_4, r_at_5, mrr_score = compute_metrics_np(logits_whole.detach().cpu().numpy(), validation_labels.numpy())
                    
                print(f"{ds} R@1: {r_at_1}, R@2: {r_at_2}, R@3: {r_at_3}, R@4: {r_at_4}, R@5: {r_at_5}, MRR: {mrr_score}")
                wandb.log({f"{ds} R@1": r_at_1, f"{ds} R@2": r_at_2, f"{ds} R@3": r_at_3, f"{ds} R@4": r_at_4, f"{ds} R@5": r_at_5, f"{ds} MRR": mrr_score}) 
            total_loss_Avg =tol_loss / len(validation_dataset)
            early_stopping(total_loss_Avg, model)
            print(f'Epoch [{epoch+1}/{args.num_epochs}], Validation Loss: {tol_loss / len(train_dataset):.4f}')
            wandb.log({'{ds} loss': tol_loss / len(train_dataset)})
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                early_stopping.save_checkpoint(total_loss_Avg, model)
                break  # Break out of the training loop

        # Check if the file exists
        if not os.path.exists(path_alvis):
            # File does not exist, so save the file.
            # For example, if you're saving a PyTorch model:
            torch.save(model.state_dict(), path_alvis)
            print(f"File saved to: {path_alvis}")
        else:
            print(f"File already exists at: {path_alvis}")


        # evaluation
        model.load_state_dict(torch.load(path_alvis))
        model.eval()
        with torch.no_grad():

            pred_whole, logits_whole = [], []
            for input, mask, label in test_dataloader:
                
                input = input.to(args.device)
                label = label.to(args.device)

                if not args.gyro:
                    b, t, c = input.shape
                    indices = np.array([range(i, i+3) for i in range(0, c, 6)]).flatten()
                    input = input[:,:,indices]

                b, t, c = input.shape
                if args.stft:
                    input_stft = input.permute(0,2,1).reshape(b * c,t)
                    input_stft = torch.abs(torch.stft(input_stft, n_fft = 25, hop_length = 28, onesided = False, center = True, return_complex = True))
                    input_stft = input_stft.reshape(b, c, input_stft.shape[-2], input_stft.shape[-1]).reshape(b, c, t).permute(0,2,1)
                    input = torch.cat((input, input_stft), dim=-1)

                input = input.reshape(b, t, 22, -1).permute(0, 3, 1, 2).unsqueeze(-1)

                logits_per_imu = model.classifier(input)
                logits_whole.append(logits_per_imu)
                
                pred = torch.argmax(logits_per_imu, dim=-1).detach().cpu().numpy()
                pred_whole.append(pred)

            pred = np.concatenate(pred_whole)
            acc = accuracy_score(test_labels, pred)
            prec = precision_score(test_labels, pred, average='macro')
            rec = recall_score(test_labels, pred, average='macro')
            f1 = f1_score(test_labels, pred, average='weighted')

            print(f"{ds} acc: {acc}, {ds} prec: {prec}, {ds} rec: {rec}, {ds} f1: {f1}")
            wandb.log({f"{ds} acc": acc, f"{ds} prec": prec, f"{ds} rec": rec, f"{ds} f1": f1})

            logits_whole = torch.cat(logits_whole)
            r_at_1, r_at_2, r_at_3, r_at_4, r_at_5, mrr_score = compute_metrics_np(logits_whole.detach().cpu().numpy(), test_labels.numpy())
                
            print(f"{ds} R@1: {r_at_1}, R@2: {r_at_2}, R@3: {r_at_3}, R@4: {r_at_4}, R@5: {r_at_5}, MRR: {mrr_score}")
            wandb.log({f"{ds} R@1": r_at_1, f"{ds} R@2": r_at_2, f"{ds} R@3": r_at_3, f"{ds} R@4": r_at_4, f"{ds} R@5": r_at_5, f"{ds} MRR": mrr_score}) 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Unified Pre-trained Motion Time Series Model')

    # model 
    parser.add_argument('--mode', type=str, default='full', choices=['random','probe','full'], help='full fine-tuning, linear probe, random init')

    # data
    parser.add_argument('--padding_size', type=int, default='200', help='padding size (default: 200)')
    parser.add_argument('--k', type=int, help='few shot samples per class (default: None)')
    parser.add_argument('--data_path', type=str, default='/home/calatrava/Documents/PhD/Thesis', help='/path/to/data/')

    # training
    parser.add_argument('--stage', type=str, default='finetune', help='training stage')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of fine-tuning epochs (default: 200)')
    parser.add_argument('--run_tag', type=str, default='exp0', help='logging tag')
    parser.add_argument('--gyro', type=int, default=0, help='using gyro or not')
    parser.add_argument('--stft', type=int, default=0, help='using stft or not')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--device', type=str, default='cuda:1', help='logging tag')
    parser.add_argument('--dataset', type=str, default='REALDISP')
    parser.add_argument('--experiment', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--TypeTest', type=str, default='self')

    

    parser.add_argument('--checkpoint', type=str, default='./checkpoint/UniMTS.pth', help='/path/to/checkpoint/')
    parser.add_argument('--SensorPosition', type=str, default='RUA')

    
    args = parser.parse_args()

    seed_all(args.seed)

    main(args)
