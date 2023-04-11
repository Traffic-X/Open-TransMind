# !/usr/bin/env python3
import numpy as np
from collections import OrderedDict

from data.util.tokenizer import *
from modeling.backbones.vit_retrieval import CLIP
from fastreid.data.data_utils import read_image
from data.transforms.build import build_transforms_lazy


def build_model(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length,
                vocab_size, transformer_width, transformer_heads, transformer_layers, qkv_bias, pre_norm,
                proj, patch_bias):

    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        qkv_bias=qkv_bias,
        pre_norm=pre_norm,
        proj=proj,
        patch_bias=patch_bias
    )

    return model


def load_pretrained(model, pretrained):
    state_dict = paddle.load(pretrained)['model']
    new_state_dict = {}
    for k in state_dict:
        new_k = k[9:] # remove 'backbones.'
        new_state_dict[new_k] = state_dict[k]
    model.set_state_dict(new_state_dict)
    return model


def get_transforms():
    # Data Preprocessing
    from paddle.vision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    # from passl.datasets.preprocess.transforms import ToRGB

    preprocess = Compose([Resize([224,224]),
                        ToTensor(),
                        ])
    image_mean = paddle.to_tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
    image_std = paddle.to_tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])

    Transforms = build_transforms_lazy(
                        is_train=False,
                        size_test=[224, 224],
                        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                    )
    return Transforms


def get_images_texts(images_root, texts_root, transform_ops):
    images = []
    texts = []

    N = 1000000

    text_dic = {}
    dic_name_text = {}
    f = open(texts_root, 'r')
    for line in f.readlines():
        line = line.strip('')
        items = line.split('$')
        name, text, wenben = items
        text_dic[name] = text
        dic_name_text[items[0].split('.')[0]] = wenben

    idx_dic = {}
    cnts = 0
    images_names = os.listdir(images_root)
    images_names.sort()
    # images_names = images_names[:N]
    # print(images_names)
    for image_name in images_names:
        only_name = image_name.split('.')[0]
        if dic_name_text.get(only_name, -1) == -1: continue
        # if 'vehicle' in only_name: continue
        img = read_image(os.path.join(images_root, image_name), "RGB")
        img = transform_ops(img)
        images.append(img)

        texts.append(dic_name_text[only_name])

        idx_dic[cnts] = text_dic[image_name]
        cnts += 1
    images = images[:N]
    texts = texts[:N]

    return images, texts, idx_dic


def get_similarity(images, texts):

    tokenizer = SimpleTokenizer()
    n = len(images)
    stage = 1000
    image_features_list = []
    text_features_list = []
    for i in range(0, n, stage):
        print(f'{i}/{n}')
        start = i
        end = min(n, i + stage)
        image_input = paddle.to_tensor(images[start: end])

        text_tokens = [tokenizer.encode("" + desc) for desc in texts[start: end]]
        text_input = paddle.zeros((len(text_tokens), 77), dtype="int64")
        sot_token = tokenizer.encoder['<|startoftext|>']
        eot_token = tokenizer.encoder['<|endoftext|>']
        for i, tokens in enumerate(text_tokens):
            tokens = [sot_token] + tokens + [eot_token]
            text_input[i, :len(tokens)] = paddle.to_tensor(tokens)

        with paddle.no_grad():
            import numpy as np
            image_input = paddle.to_tensor(image_input)
            text_input = paddle.to_tensor(text_input)
            image_features = model.encode_image(image_input)
            image_features_list.append(image_features)
            text_features = model.encode_text(text_input)
            text_features_list.append(text_features)
 
    image_features = paddle.concat(image_features_list, axis=0)
    text_features = paddle.concat(text_features_list, axis=0)

    image_features /= image_features.norm(axis=-1, keepdim=True)
    text_features /= text_features.norm(axis=-1, keepdim=True)
    print('image_features.shape', image_features.shape)
    print('text_features.shape', text_features.shape)
    similarity = paddle.matmul(text_features, image_features.t()).cpu().numpy()
    return similarity


def eval(similarity, idx_dic):
    similarity_argsort = np.argsort(-similarity, axis=1)
    print('similarity_argsort.shape', similarity_argsort.shape)

    def func_check_same_attri(s1, s2):
        s1 = s1.split(',')
        s2 = s2.split(',')
        if len(s1) != 21: return 0
        if s1[0] != s2[0]:
            return 0
        same = 0
        sum1 = 0
        for i in range(len(s1)):
            if s1[i] == '1' and s2[i] == '1':
                same += 1
            if s1[i] == '1': sum1 += 1
        # print(same, sum1, same==sum1)
        return same == sum1

    tp = 0
    map = 0
    topk = 10
    for i in range (len(similarity_argsort)):
        # print(i, similarity_argsort[i][:topk])
        get = 1e-6
        ap = 0
        for j in range(topk):
            # print(i, similarity_argsort[i][j], idx_dic[i], idx_dic[similarity_argsort[i][j]], func_check_same_attri(idx_dic[i], idx_dic[similarity_argsort[i][j]]))
            if (idx_dic[i] == idx_dic[similarity_argsort[i][j]]) or func_check_same_attri(idx_dic[i], idx_dic[similarity_argsort[i][j]]):
                get += 1
                # print(get, j + 1, similarity_argsort[i][j])
                ap += (get / (j+1))
        ap /= get
        map += ap
    print('score:', map / len(similarity_argsort))


if __name__ == '__main__':

    pretrained = "./outputs/vitbase_retrieval/model_final.pdmodel"
    images_root = './datasets/val/val_images/'
    texts_root = './datasets/val/val_label.txt' 


    model = build_model(embed_dim=512,
                        image_resolution=224,
                        vision_layers=12,
                        vision_width=768,
                        vision_patch_size=32,
                        context_length=77,
                        vocab_size=49408,
                        transformer_width=512,
                        transformer_heads=8,
                        transformer_layers=12,
                        qkv_bias=True,
                        pre_norm=True,
                        proj=True,
                        patch_bias=False)

    model = load_pretrained(model, pretrained)

    transform_ops = get_transforms()

    images, texts, idx_dic = get_images_texts(images_root, texts_root, transform_ops)

    similarity = get_similarity(images, texts)

    eval(similarity, idx_dic)