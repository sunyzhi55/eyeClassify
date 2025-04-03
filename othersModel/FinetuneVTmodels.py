# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from othersModel.MIL_VT import *

def rename_pretrain_weight(checkpoint):
    state_dict_old = checkpoint['state_dict']
    state_dict_new = OrderedDict()
    for key, value in state_dict_old.items():
        state_dict_new[key[len('module.'):]] = value
    return state_dict_new


#LoadTongrenPretrainedWeight_NoDistillation
def MIL_VT_FineTune(image_size=384,
                    MODEL_PATH_finetune = 'weights/fundus_pretrained_VT_small_patch16_384_5Class.pth.tar',
                    num_classes=5):
    """Load pretrain weight from distillation model, to train a plain model"""

    # model = create_model(model_name=base_model,
    #         pretrained=False,
    #         num_classes=num_classes,
    #         drop_rate=0,
    #         drop_path_rate=0.1,
    #         drop_block_rate=None,
    #     )
    model = MIL_VT_small_patch16(
        image_size=image_size,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0,
        drop_path_rate=0.1,
        # drop_block_rate=None,
        )

    checkpoint0 = torch.load(MODEL_PATH_finetune, map_location='cpu')
    checkpoint_model = rename_pretrain_weight(checkpoint0)

    state_dict = model.state_dict()
    checkpoint_keys = list(checkpoint_model.keys())
    for tempKey in list(state_dict.keys()):
        if tempKey not in checkpoint_keys:
            print('Missing Key not in pretrain model: ', tempKey)


    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model: # and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    num_extra_tokens_chechpoint = 2
    print('pos_embed: ', embedding_size, num_patches, num_extra_tokens)

 
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens_chechpoint) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens_chechpoint:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint_model, strict=False)

    return model


class MIL_VIT_Double(nn.Module):
    def __init__(self, image_size=384,
                 MODEL_PATH_finetune = 'weights/fundus_pretrained_VT_small_patch16_384_5Class.pth.tar',
                 num_classes=7):
        super(MIL_VIT_Double, self).__init__()
        self.model1 = MIL_VT_FineTune(image_size=image_size, MODEL_PATH_finetune = MODEL_PATH_finetune, num_classes=num_classes)
        # self.model2 = MIL_VT_FineTune(image_size=image_size, MODEL_PATH_finetune = MODEL_PATH_finetune, num_classes=num_classes)
        self.fc = nn.Sequential(
            nn.Linear(2 * num_classes, 10),
            nn.ReLU(),
            nn.Linear(10, num_classes),
        )

    def forward(self, x, y):
        if self.training:
            outputs_class1, outputs_MIL1 = self.model1(x)
            outputs_class2, outputs_MIL2  = self.model1(y)
            fusion = torch.cat((outputs_class1, outputs_class2), dim=1)
            result = self.fc(fusion)
            return outputs_class1, outputs_MIL1, outputs_class2, outputs_MIL2, result
        else:
            outputs_class1 = self.model1(x)
            outputs_class2  = self.model1(y)
            fusion = torch.cat((outputs_class1, outputs_class2), dim=1)
            result = self.fc(fusion)
            return outputs_class1, outputs_class2, result



# if __name__ == '__main__':
#     img_size = 384
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     MODEL_PATH_finetune = '/data4/wangchangmiao/shenxy/PretrainedModel/fundus_pretrained_VT_small_patch16_384_5Class.pth.tar'
#     model = MIL_VT_FineTune(image_size=img_size, MODEL_PATH_finetune = MODEL_PATH_finetune, num_classes=5).to(device)
#     print(model)
#     # 打印模型参数量
#     print("model parameters:", sum(p.numel() for p in model.parameters()))
#     x = torch.randn(1, 3, img_size, img_size).to(device)
#     output1, output2 = model(x)
#     print("output1", output1.shape)
#     print("output2", output2.shape)

