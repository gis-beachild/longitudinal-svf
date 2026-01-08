from torchdiffeq import odeint_adjoint as odeint
from Utils.Utls import *
from Utils.Loss import *
import os
import random
import time
from Network.DynamicNet import DynamicNet
import argparse
import yaml




def main(args):
    with open(args.dataset_yaml, "r") as f:
        config = yaml.safe_load(f)
    csize = config['csize']
    name = config['name']
    num_classes = config['num_classes']
    date_format = config['date_format']
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed) #CPU随机种子确定
    torch.cuda.manual_seed(seed) #GPU随机种子确定
    torch.cuda.manual_seed_all(seed) #所有的GPU设置种子
    torch.backends.cudnn.benchmark = False #模型卷积层预先优化关闭
    torch.backends.cudnn.deterministic = True #确定为默认卷积算法
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    subject_path = name
    # 模型的保存路径
    savePath = os.path.join(args.savePath, "noder", subject_path)
    os.makedirs(os.path.join(savePath), exist_ok=True)

    imgs, segs, times = load_imgs_and_time(config['csv_path'], padding=csize, num_classes=num_classes, date_format=date_format)


    print(len(imgs))
    print(imgs[0].shape)
    print(np.min(imgs[0]))
    print(np.max(imgs[0]))
    print(times)

    # 示例subject002_S_4229的序列长度为9，这里我们按照  训练:测试=7:2 的比例 进行划分（默认是80%：20%，可根据不同subject进行调整）
    train_List = imgs
    train_list_seg = segs
    train_times = times
    seq_length = len(imgs)
    im_shape = train_List[0].shape
    im_shape = (im_shape[-3], im_shape[-2], im_shape[-1])
    train_List = [torch.from_numpy(img).to(device).float().unsqueeze(0).unsqueeze(0) for img in train_List]
    train_list_seg = [torch.from_numpy(img).to(device).float().unsqueeze(0) for img in train_list_seg]
    # 定义网络v(这里采用简化的版本)
    Network = DynamicNet(img_sz=im_shape,
                         smoothing_kernel='AK',
                         smoothing_win=15,
                         smoothing_pass=1,
                         ds=2,
                         bs=32
                         ).to(device)
    optimizer = torch.optim.Adam(Network.parameters(), lr=0.001, amsgrad=True)
    epoches = args.epochs
    scale_factor = torch.tensor(im_shape).to(device).view(1, 3, 1, 1, 1) * 1.
    ST = SpatialTransformer(im_shape).to(device)  # spatial transformer to warp image
    grid = generate_grid3D_tensor(im_shape).unsqueeze(0).to(device)  # [-1,1] 1*3*144*176*144 (identity map)
    total_record = []
    loss_NCC = NCC(win=21)
    for i in range(epoches):
        # 开始计时
        s_t = time.time()
        all_phi = odeint(func=Network, y0=grid, t=torch.tensor(train_times).to(device), method="rk4", rtol=1e-3,
                         atol=1e-5).to(device)
        all_v = all_phi[1:] - all_phi[:-1]
        all_phi = (all_phi + 1.) / 2. * scale_factor  # [-1, 1] -> voxel spacing  恢复到标准的坐标系
        grid_voxel = (grid + 1.) / 2. * scale_factor  # [-1, 1] -> voxel spacing
        # 记录各种loss
        total_loss = 0.0
        epoch_loss = []
        epoch_loss_NCC = []
        epoch_loss_MSE = []
        epoch_loss_v = []
        epoch_loss_J = []
        epoch_folding = []
        epoch_loss_df = []
        epoch_loss_bdr = []


        # 对每一个时间点的预测进行loss计算
        for n in range(1, seq_length):
            phi = all_phi[n]
            df = phi - grid_voxel  # with grid -> without grid（此处的df是offset）
            warped_moving, df_with_grid = ST(train_List[0], df, return_phi=True)
            warped_moving_seg, df_with_grid = ST(train_list_seg[0], df, return_phi=True)

            # similarity loss（NCC）
            loss_sim = loss_NCC(warped_moving, train_List[n])
            epoch_loss_NCC.append(loss_sim.clone().detach().cpu())

            # loss_ncc = loss_NCC(warped_moving,train_List[n])
            # epoch_loss_NCC.append(loss_ncc.clone().detach().cpu())

            loss_mse = MSE(warped_moving_seg, train_list_seg[n])
            epoch_loss_MSE.append(loss_mse.clone().detach().cpu())

            warped_moving = warped_moving.squeeze(0).squeeze(0)
            # V magnitude loss
            loss_v = 0.00005 * magnitude_loss(all_v)
            epoch_loss_v.append(loss_v.clone().detach().cpu())
            # neg Jacobian loss
            loss_J = 0.000001 * neg_Jdet_loss1(df_with_grid)
            epoch_loss_J.append(loss_J.clone().detach().cpu())

            # folding
            folding = calculate_folding(df, device)
            epoch_folding.append(folding)

            # phi dphi/dx loss
            loss_df = args.lambdaGrad * smoothloss_loss(df)
            epoch_loss_df.append(loss_df.clone().detach().cpu())
            # bdr loss
            loss_bdr = 0.001 * boundary_loss(df, img_size=csize)
            epoch_loss_bdr.append(loss_bdr.clone().detach().cpu())
            # 各项loss求和
            loss = loss_mse + loss_df + loss_bdr
            # + loss_v + loss_J + loss_df
            epoch_loss.append(loss.clone().detach().cpu())
            # 各个时间点的loss求和
            total_loss = total_loss + loss

        optimizer.zero_grad()
        total_loss = total_loss / (seq_length - 1)
        total_loss.backward()
        optimizer.step()

        # 结束计时
        e_t = time.time()

        print(
            "Iteration: {0} loss_NCC: {1:.3e}  loss_v: {2:.3e} loss_J: {3:.3e} loss_df: {4:.3e} total_loss: {5:.3e} time_cost: {6:.3e} loss_MSE: {7:.3e} loss_bdr: {8:.3e} folding: {9:.3e}"
            .format(i + 1,
                    np.mean(epoch_loss_NCC),
                    np.mean(epoch_loss_v),
                    np.mean(epoch_loss_J),
                    np.mean(epoch_loss_df),
                    total_loss.item(),
                    e_t - s_t,
                    np.mean(epoch_loss_MSE),
                    np.mean(epoch_loss_bdr),
                    np.mean(epoch_folding)
                    )
            )
        # 保存每个epoch的记录
        epoch_record = {"Iteration": i + 1,
                        "loss_NCC": np.mean(epoch_loss_NCC),
                        "loss_v": np.mean(epoch_loss_v),
                        "loss_J": np.mean(epoch_loss_J),
                        "folding": np.mean(epoch_folding),
                        "loss_df": np.mean(epoch_loss_df),
                        "loss_bdr": np.mean(epoch_loss_bdr),
                        "total_loss": total_loss.item(),
                        "time_cost": e_t - s_t}
        total_record.append(epoch_record)
        # 每50个epoch保存一次模型
        if (i + 1) % 50 == 0:
            torch.save(Network.state_dict(), os.path.join(savePath, "model.pkl"))



if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--dataset_yaml', type=str, help='Subject name', default="/home/florian/Documents/Programs/longitudinal-svf-pair/src/configs/data/atlasimages.yaml")
    argparse.add_argument('--savePath', type=str, default="./model-save", help='Model save path')
    argparse.add_argument('--lambdaGrad', type=float, default=0.005, help='Lambda for gradient loss')
    argparse.add_argument('--epochs', type=int, default=3000, help='Number of epochs')
    args = argparse.parse_args()
    main(args)

