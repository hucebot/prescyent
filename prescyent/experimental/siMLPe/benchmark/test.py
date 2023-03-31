import argparse
from scipy.spatial.transform import Rotation as R

import numpy as np

from prescyent.predictor import AutoPredictor
from prescyent.dataset import H36MDataset, H36MDatasetConfig
from prescyent.predictor.constant_predictor import ConstantPredictor

from config import config
from h36m_eval import H36MEval

import torch
from torch.utils.data import DataLoader


results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']


def regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36):
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)

    for (motion_input, motion_target) in pbar:
        b,n,c,_ = motion_input.shape
        num_samples += b
        motion_input = motion_input.reshape(b, n, 32, 3)
        motion_input = motion_input[:, :, joint_used_xyz]
        outputs = []
        step = config.motion.h36m_target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():
                motion_input_ = motion_input.clone()
                output = model.predict(motion_input_, step)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        motion_pred = torch.cat(outputs, axis=1)[:,:25]

        motion_target = motion_target.detach()
        b,n,c,_ = motion_target.shape

        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()
        pred_rot = motion_pred.clone().reshape(b,n,22,3)
        motion_pred = motion_target.clone().reshape(b,n,32,3)
        motion_pred[:, :, joint_used_xyz] = pred_rot

        tmp = motion_gt.clone()
        tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]
        motion_pred = tmp
        motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36

def test(config, model, dataloader) :

    m_p3d_h36 = np.zeros([config.motion.h36m_target_length])
    titles = np.array(range(config.motion.h36m_target_length)) + 1
    joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
    num_samples = 0

    pbar = dataloader
    m_p3d_h36 = regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36)

    ret = {}
    for j in range(config.motion.h36m_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return [round(ret[key][0], 1) for key in results_keys]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # mp = "/home/abiver-local/repositories/HuCeBot/prescyent/data/models/h36m/50_10/MlpPredictor/version_27"
    parser.add_argument('--model_path', type=str, help='=model path')
    parser.add_argument('--do_constant', action="store_true", help='=use constant predictor')
    args = parser.parse_args()
    if args.do_constant:
        model = ConstantPredictor("ConstantPredictor/")
    else:
        model = AutoPredictor.load_from_config(args.model_path)

    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    dataset = H36MEval(config, 'test')

    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=1, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)

    print(test(config, model, dataloader))
    print("num_params: %.3f" % (sum(p.numel() for p in model.model.torch_model.parameters()) / 1000000))
    print("num_trainable_params: %.3f" % (sum(p.numel() for p in model.model.torch_model.parameters() if p.requires_grad) / 1000000))
