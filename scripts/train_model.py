import time
import sys
import os
import yaml
from sklearn.metrics import f1_score
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import torch, gc
from Project.utils.face_dataset import FaceImageDataset
from Project.models.gender_age_model import GenderAgeModel

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def main():
    gc.collect()
    torch.cuda.empty_cache()

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/hyper_parameters.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    print(torch.__version__)  # PyTorch 버전 확인
    print(torch.version.cuda)  # CUDA 버전 확인
    print(torch.cuda.is_available())  # True가 나오면 GPU가 사용 가능한 상태
    print(torch.cuda.device_count())  # 사용 가능한 GPU 수
    print(torch.cuda.current_device())  # 현재 사용 중인 GPU
    print(torch.cuda.get_device_name(torch.cuda.current_device()))  # GPU 이름
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomCrop(224, padding=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = FaceImageDataset('../data/Training', transform=dataset_transform)
    val_data = FaceImageDataset('../data/Validation', transform=dataset_transform)

    train_loader = torch.utils.data.DataLoader(train_data, params['batch_size'], shuffle=True)
    dev_loader = torch.utils.data.DataLoader(val_data, params['batch_size'], shuffle=False)

    model = GenderAgeModel().to(device)

    n = sum(p.numel() for p in model.parameters())
    print(f'The Number of Parameters of {model.model_name}: {n:,}')
    print(params)

    criterion_gender = nn.CrossEntropyLoss()
    criterion_age = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['l2_reg_lambda'])
    decay_step = [20000, 32000]
    step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_step, gamma=0.1)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath((os.path.join("runs", timestamp)))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    summary_dir = os.path.join(out_dir, "summaries")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    writer = SummaryWriter(summary_dir)

    start_time = time.time()
    lowest_val_loss = float('inf')
    global_steps = 0
    train_age_errors = []
    print('========================================')
    print("Training...")
    for epoch in range(params['max_epochs']):
        train_loss = 0
        train_correct_cnt = 0
        train_batch_cnt = 0
        model.train()
        all_gender_preds = []
        all_gender_labels = []
        for img, gender, age in train_loader:
            img = img.to(device)
            gender = gender.to(device)
            age = age.float().to(device)

            output_gender, output_age = model(img)
            output_age = output_age.squeeze()

            loss_gender = criterion_gender(output_gender, gender)
            loss_age = criterion_age(output_age, age)
            loss = loss_age + loss_gender

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_lr_scheduler.step()

            train_loss += loss.item()
            train_batch_cnt += 1

            _, gender_pred = torch.max(output_gender, 1)
            gender_pred = gender_pred.squeeze()
            train_correct_cnt += int(torch.sum(gender_pred == gender))

            age_errors = torch.abs(output_age.squeeze() - age)
            train_age_errors.append(age_errors.detach())

            all_gender_preds.extend(gender_pred.cpu().numpy())
            all_gender_labels.extend(gender.cpu().numpy())

            batch_total = gender.size(0)
            batch_correct = int(torch.sum(gender_pred == gender))
            batch_gender_acc = batch_correct / batch_total

            writer.add_scalar("Batch/Loss", loss.item(), global_steps)
            writer.add_scalar("Batch/Acc", batch_gender_acc, global_steps)
            writer.add_scalar("LR/Learning_rate", step_lr_scheduler.get_last_lr()[0], global_steps)

            global_steps += 1
            if global_steps % 100 == 0:
                print('Epoch [{}], Step [{}], Loss: {:.4f}'.format(epoch+1, global_steps, loss.item()))

        train_gender_acc = train_correct_cnt / len(train_data) * 100
        train_f1 = f1_score(all_gender_labels, all_gender_preds) * 100
        train_mae_age = torch.cat(train_age_errors).mean()
        train_ave_loss = train_loss / train_batch_cnt
        training_time = (time.time() - start_time) / 60
        writer.add_scalar("Train/Loss", train_ave_loss, epoch)
        writer.add_scalar("Train/Gender Acc", train_gender_acc, epoch)
        writer.add_scalar("Train/F1 Score", train_f1, epoch)
        writer.add_scalar("Epoch/Train MAE Age", train_mae_age.item(), epoch)
        print('========================================')
        print("epoch:", epoch + 1, "/ global_steps:", global_steps)
        print("training dataset average loss: %.3f" % train_ave_loss)
        print("training_time: %.2f minutes" % training_time)
        print("learning rate: %.6f" % step_lr_scheduler.get_last_lr()[0])

        val_correct_cnt = 0
        val_loss = 0
        val_batch_cnt = 0
        val_age_errors = []
        all_val_gender_preds = []
        all_val_gender_labels = []
        print('========================================')
        print('Validation...')
        model.eval()
        with torch.no_grad():
            for img, gender, age in dev_loader:
                img = img.to(device)
                gender = gender.to(device)
                age = age.float().to(device)

                output_gender, output_age = model(img)
                output_age = output_age.squeeze()

                loss_gender = criterion_gender(output_gender, gender)
                loss_age = criterion_age(output_age, age)
                loss = loss_gender + loss_age

                val_loss += loss.item()
                val_batch_cnt += 1
                _, gender_pred = torch.max(output_gender, 1)
                gender_pred = gender_pred.squeeze()
                val_correct_cnt += int(torch.sum(gender_pred == gender))

                age_errors = torch.abs(output_age.squeeze() - age)
                val_age_errors.append(age_errors)

                all_val_gender_preds.extend(gender_pred.cpu().numpy())
                all_val_gender_labels.extend(gender.cpu().numpy())

        val_gender_acc = val_correct_cnt / len(val_data) * 100
        val_f1 = f1_score(all_val_gender_labels, all_val_gender_preds) * 100
        val_mae_age = torch.cat(val_age_errors).mean()
        val_ave_loss = val_loss / val_batch_cnt

        print("validation dataset gender accuracy: %.2f" % val_gender_acc)
        # print("validation dataset f1 score: %.2f" % val_f1)
        print("validation dataset age MAE: %.2f" % val_mae_age)
        print('========================================')

        writer.add_scalar("Val/Gender Acc", val_gender_acc, epoch)
        writer.add_scalar("Val/F1 Score", val_f1, epoch)
        writer.add_scalar("Val/Age Loss", val_ave_loss, epoch)


        if val_ave_loss < lowest_val_loss:
            save_path = checkpoint_dir + '/epoch_' + str(epoch + 1) + '.pth'
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)

            save_path = checkpoint_dir + '/best.pth'
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)
            lowest_val_loss = val_ave_loss


if __name__ == '__main__':
    main()
