import numpy as np
import torch
import torch.nn.functional as F

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

def main(args):

    repo_id = "xiyuanz/UniMTS"
    checkpoint_file = "checkpoint/UniMTS.pth"
    config_file = "config.json"
    data_file = "UniMTS_data.zip"

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
    train_dataloader_list, test_dataloader_list = [], []
    for real_inputs, real_masks, real_labels in zip(train_inputs_list, train_masks_list, train_labels_list):
        train_dataset = TensorDataset(real_inputs, real_masks, real_labels)
        train_dataloader_list.append(DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True))
    for real_inputs, real_masks, real_labels in zip(test_inputs_list, test_masks_list, test_labels_list):
        test_dataset = TensorDataset(real_inputs, real_masks, real_labels)
        test_dataloader_list.append(DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False))

    date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
    wandb.init(
        project='UniMTS',
        name=f"{args.run_tag}_{args.stage}_{args.mode}_k={args.k}_" + f"{date}" 
    )

    save_path = './checkpoint/%s/' % args.run_tag
    os.makedirs(save_path, exist_ok=True)

    for ds, train_dataloader, test_dataloader, test_labels, label_list, all_text, num_class in \
            zip(dataset_list, train_dataloader_list, test_dataloader_list, test_labels_list, label_list_list, all_text_list, num_classes_list):
        
        args.num_class = num_class
        model = ContrastiveModule(args).cuda()
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

                input = input.cuda()
                labels = label.cuda()

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

            if best_loss is None or tol_loss < best_loss:
                best_loss = tol_loss
                torch.save(model.state_dict(), os.path.join(save_path, f'{ds}_k={args.k}_best_loss.pth'))

        # evaluation
        model.load_state_dict(torch.load(os.path.join(save_path, f'{ds}_k={args.k}_best_loss.pth')))
        model.eval()
        with torch.no_grad():

            pred_whole, logits_whole = [], []
            for input, mask, label in test_dataloader:
                
                input = input.cuda()
                label = label.cuda()

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
            f1 = f1_score(test_labels, pred, average='macro')

            print(f"{ds} acc: {acc}, {ds} prec: {prec}, {ds} rec: {rec}, {ds} f1: {f1}")
            wandb.log({f"{ds} acc": acc, f"{ds} prec": prec, f"{ds} rec": rec, f"{ds} f1": f1})

            logits_whole = torch.cat(logits_whole)
            r_at_1, r_at_2, r_at_3, r_at_4, r_at_5, mrr_score = compute_metrics_np(logits_whole.detach().cpu().numpy(), test_labels.numpy())
                
            print(f"{ds} R@1: {r_at_1}, R@2: {r_at_2}, R@3: {r_at_3}, R@4: {r_at_4}, R@5: {r_at_5}, MRR: {mrr_score}")
            wandb.log({f"{ds} R@1": r_at_1, f"{ds} R@2": r_at_2, f"{ds} R@3": r_at_3, f"{ds} R@4": r_at_4, f"{ds} R@5": r_at_5, f"{ds} MRR": mrr_score}) 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Unified Pre-trained Motion Time Series Model')

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
    parser.add_argument('--TypeTest', type=str, default='self')
    

    parser.add_argument('--checkpoint', type=str, default='./checkpoint/UniMTS.pth', help='/path/to/checkpoint/')
    parser.add_argument('--SensorPosition', type=str, default='RUA')
    
    args = parser.parse_args()

    main(args)
