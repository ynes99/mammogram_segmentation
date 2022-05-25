import cv2
from tensorflow.python.framework.ops import disable_eager_execution
from .forms import *
from os import mkdir
import os.path
from .Traitement_image.Crop_From_Center import crop_image_from_center
from .Algorithms.FCM_algorithm.FCM import FCM
from .Algorithms.EM_algorithm.EM import *
from .utilities.Path_Join import p_join
from .utilities.IOU import iou
from .utilities.Dice_coef import dice_coef
import matplotlib.image as mpimg
from pathlib import Path
from .models import UploadedImage
import pandas as pd
from django.shortcuts import render
from .Algorithms.U_net_algorithm.data import *
from .Algorithms.U_net_algorithm.model import *
import numpy as np

disable_eager_execution()


# Create your views here.

def iou_for_cnn(center, shape, gt, resized_gt, predict_img):
    # Get the scaling factor
    img_shape = gt.shape
    reshaped_img_shape = (256, 256)
    scale = np.flipud(np.divide(reshaped_img_shape,
                                img_shape))
    # you have to flip because the image.shape is (y,x) but your corner points are (x,y)

    # use this on to get new top left corner and bottom right corner coordinates
    new_center = int(np.multiply(center, scale)[0]), int(np.multiply(center, scale)[1])
    new_shape = int(np.multiply(shape, scale)[0] + 20), int(np.multiply(shape, scale)[1] + 20)

    y_true = crop_image_from_center(resized_gt, shape=new_shape, center=new_center)
    y_pred = crop_image_from_center(predict_img, shape=new_shape, center=new_center)
    intersection = 0
    union = new_shape[0] * new_shape[1]
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            if y_true[i][j] == y_pred[i][j]:
                intersection += 1
    print(f'IOU : {(intersection / union)}')
    return intersection / union


def image_prep(n, preproc):
    number = n
    df = pd.read_excel(r"C:\Users\Asus\Desktop\PCD\Organized dataset\metadata\metadata_good2.0.xlsx")
    if preproc:
        parent_dir = r'C:\Users\Asus\Desktop\PCD\Organized dataset\preprocessed_dataset'
        center = eval(df['center of roi preprocessed'][number])
        g_path = os.path.join(df['file path'][number], r"GT_mask.jpg")
    else:
        parent_dir = r'C:\Users\Asus\Desktop\PCD\Organized dataset\curated'
        center = eval(df['center of roi'][number])
        g_path = df['ROI mask file path'][number]
    shape = eval(df['Shape of cropped image'][number])
    image_path = p_join(parent_dir, df['image full mammo path'][number])
    gt = mpimg.imread(p_join(parent_dir, g_path))
    return g_path, gt, image_path, center, shape


def save_res(path, result, gt, center, shape):
    cv2.imwrite(path, result)
    gt_crop = crop_image_from_center(gt, center, shape)
    list_path = path.split('\\')
    list_path.pop()
    path_gt = '\\'.join(list_path)

    path_gt_cropped = p_join(path_gt, 'gt_mask_cropped.jpg')
    path_gt = p_join(path_gt, 'gt_mask.jpg')

    cv2.imwrite(path_gt_cropped, gt_crop)
    cv2.imwrite(path_gt, gt)

    iou_coef = iou(gt_crop, result)
    dice = dice_coef(gt_crop, result)
    return dice, iou_coef, path_gt_cropped, path_gt


def retrive():
    instance = UploadedImage.objects.all().last()
    df = pd.read_excel(r"C:\Users\Asus\Desktop\PCD\Organized dataset\metadata\metadata_good2.0.xlsx")
    rslt_df_test = df[df['description'] == 'Mass-Test']
    rslt_df_test_final = rslt_df_test.drop_duplicates('file path')
    list_test = list(rslt_df_test_final['image full mammo path'])
    path_image = getattr(instance, 'title')
    filename = Path(path_image).stem
    mammo_path = list_test[eval(filename)]
    index_df = df[df['image full mammo path'] == mammo_path].index.values[0]
    return index_df


def check_image(image):
    corners = [image[0][0], image[0][-1], image[-1][0], image[-1][-1]]
    if corners.count(0) < 2:
        inverted = np.invert(image)
    else:
        inverted = image
    return inverted


def path_static_conv(path_showcase):
    list_path = path_showcase.split('\\')
    list_path = list_path[8::]
    path_static = '\\'.join(list_path)
    return path_static


def segment(directory):
    # __FCM__
    # --------------Retrieving Uploaded Image------------
    index_in_df = retrive()

    # --------------Prepping Image------------
    gt_path, gt, image_path, center, shape = image_prep(index_in_df, False)
    image = mpimg.imread(image_path)
    crop = preproc_and_crop(image, center, shape)

    # --------------Clustering------------
    cluster = FCM(crop, image_bit=8, n_clusters=2, m=2, epsilon=0.0001, max_iter=1000)
    cluster.form_clusters()
    result_p_fcm = cluster.result_def
    result_fcm = check_image(result_p_fcm)
    # --------------Saving Result------------

    path_fcm = p_join(directory, r'test_fcm.jpg')

    dice, iou_coef_fcm, gt_cropped_path, gt_path_res = save_res(path_fcm, result_fcm, gt, center, shape)

    # __U-NET__
    _, gt, image_path_u, center_u, shape_u = image_prep(index_in_df, True)
    # --------------Segmentation--------------
    testgene = testGenerator_1(image_path_u)
    model = unet()
    model.load_weights(r"C:\Users\Asus\PycharmProjects\pythonProject1\unet_membrane.hdf5")
    results = model.predict_generator(testgene, 1, verbose=1)
    img = results[0]
    img = img_as_ubyte(img)
    (_, predict_img) = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    print("Predicting Image Done ! (U_Net)")
    # --------------saving result----------------
    path_u_net = p_join(directory, r'test_u_net.jpg')
    cv2.imwrite(path_u_net, predict_img)

    # --------------iou coefficient--------------
    resized_gt = cv2.resize(gt, (256, 256))

    iou_coef_u_net = iou_for_cnn(center, shape, gt, resized_gt, predict_img)

    # __EM__
    # --------------Clustering------------
    gray_img = read_img(filename=image_path, center=center, shape=shape)
    x, y = gray_img.shape
    dim = (x, y)
    flat_im = flatten_img(gray_img)
    labels, means, cov, pis, likelihood_arr, means_arr = EM_cluster(flat_im, 2, error=0.001)
    means = np.array([element[0] for element in means])
    em_img = means[labels]
    result_p_em = em_img.reshape(dim[0], dim[1])
    a = result_p_em.min()
    ret, im_bw = cv2.threshold(result_p_em, a + 1, 255, cv2.THRESH_BINARY)
    # --------------Saving Result------------
    path_em = p_join(directory, r'test_em.jpg')
    result_em = check_image(im_bw)
    dice, iou_coef_em, _, _ = save_res(path_em, result_em, gt, center, shape)

    # --------------Result--------------
    path_fcm = path_static_conv(path_fcm)
    path_em = path_static_conv(path_em)
    path_u_net = path_static_conv(path_u_net)
    gt_cropped_path = path_static_conv(gt_cropped_path)
    gt_path_res = path_static_conv(gt_path_res)
    context = [{"image": path_u_net,
                "algorithm": 'Convolutional Neural Network U-Net Architecture', 'IOU': iou_coef_u_net,
                },
               {"image": path_fcm, "algorithm": 'Fuzzy C-Means', 'IOU': iou_coef_fcm,
                },
               {"image": path_em, "algorithm": 'Expectation-Maximization', 'IOU': iou_coef_em,
                },
               ]
    return context, gt_path_res, gt_cropped_path


def image_view(request):
    alert_message = False
    res_context_dic = []
    gt_path = ''
    gt_cropped_path = ''
    if request.method == 'POST':
        submitted_form = UploadImageForm(request.POST, request.FILES)
        if submitted_form.is_valid():
            submitted_form.save()
            alert_message = {
                'status': True,
                'message': 'Successfully saved the image'
            }
            # ---------------Create directory to store results----------------
            path_dir = r'C:\Users\Asus\PycharmProjects\pythonProject1\my_site\image\static\showcase'
            ref_dir = os.listdir(path_dir)[-1]
            index_dir = ref_dir.split('_')[-1]
            tmp = Path(f'res_{eval(index_dir) + 1}')
            directory = p_join(path_dir, tmp)
            mkdir(directory)
            res_context_dic, gt_path, gt_cropped_path = segment(directory)

        else:
            alert_message = {
                'status': False,
                'message': 'Form data is invalid. Please check if your image / title is repeated'
            }

    form = UploadImageForm()
    context = {
        'alert_data': alert_message,
        'form': form,
        'images': UploadedImage.objects.all().last(),
        'results': res_context_dic,
        'gt': gt_path,
        'gt_cropped': gt_cropped_path
    }

    return render(request, 'my_app/image_upload.html', context=context)


def index(request):
    return render(request, "my_app/index.html")


def symptoms(request):
    return render(request, "my_app/symptoms.html")


def detection(request):
    return render(request, "my_app/treatment.html")
