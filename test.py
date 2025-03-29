from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from utils import get_tuple_transform_ops, tensor_to_pil


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--im_A_path", default="asset/im_A.png", type=str)
    parser.add_argument("--im_B_path", default="asset/im_B.png", type=str)

    parser.add_argument("--save_path", default="result.png", type=str)
    parser.add_argument("--checkpoint", default="asset/model.pt")
 
    args, _ = parser.parse_known_args()

    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path
    checkpoint_path = args.checkpoint
    H, W = 320, 640

    model = torch.jit.load(checkpoint_path)
    model.eval()

    im1 = Image.open(im1_path).resize((W, H))
    im2 = Image.open(im2_path).resize((W, H))

    if im1.mode == "RGBA": im1 = im1.convert("RGB")
    if im2.mode == "RGBA": im2 = im2.convert("RGB")
    test_transform =get_tuple_transform_ops(resize=(H, W), normalize=True)
    query, support = test_transform((im1, im2))
    batch_query = query[None].to(device)
    batch_support = support[None].to(device)

    warp, certainty = model.match(batch_query, batch_support, device=device, DK_erp=True)

    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    im2_transfer_rgb = F.grid_sample(
    x2[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
    )[0]
    im1_transfer_rgb = F.grid_sample(
    x1[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
    )[0]
    warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)
    white_im = torch.ones((H,2*W),device=device)

    vis_im = certainty * warp_im + (1 - certainty) * white_im
    tensor_to_pil(vis_im, unnormalize=False).save(save_path)
