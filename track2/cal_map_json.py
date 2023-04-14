import os
import json


def read_txt(file):
    text_list = []
    image_names_list = []
    f = open(file, 'r')
    for line in f.readlines():
        line = line.strip()
        items = line.split('$')
        text_list.append(items[0])
        image_names_list.append(items[1:])
    return text_list, image_names_list


def read_json(file):
    text_list = []
    image_names_list = []
    with open(file, 'r', encoding='utf-8') as f:
        j = json.load(f)
    results = j['results']
    for i in results:
        text = i['text']
        image_names = i['image_names']
        text_list.append(text)
        image_names_list.append(image_names)
    return text_list, image_names_list



def check_text(gt_list, pred_list):
    n_gt = len(gt_list)
    n_pred = len(pred_list)
    if n_gt != n_pred:
        return 0
    else:
        for i in range(n_gt):
            if gt_list[i] != pred_list[i]:
                return 0
        return 1


def calculate_map(gt, pred):
    gt_text_list, gt_image_names_list = read_txt(gt)
    pred_text_list, pred_image_names_list = read_json(pred)
    if not check_text(gt_text_list, pred_text_list):
        print('The order of texts is wrong.')
        return
    else:
        gt_image_name_dic = {}
        for i in range(len(gt_image_names_list)):
            gt_image_name_dic[i] = {}
            for j in range(len(gt_image_names_list[i])):
                img_name = gt_image_names_list[i][j]
                gt_image_name_dic[i][img_name] = 1
        
        map = 0
        for i in range(len(pred_image_names_list)):
            get = 1e-6
            ap = 0
            for j in range(len(pred_image_names_list[i])):
                img_name = pred_image_names_list[i][j]
                if gt_image_name_dic[i].get(img_name, -1) != -1:
                    get += 1
                    # print(get, j + 1, similarity_argsort[i][j])
                    ap += (get / (j+1))
            ap /= get
            map += ap
        print('score:', map / len(pred_image_names_list))


if __name__ == '__main__':
    gt = '/root/paddlejob/workspace/env_run/datasets/textimage/person_car_bk/0328/test_gt.txt'
    pred = 'infer_json.json'
    calculate_map(gt, pred)