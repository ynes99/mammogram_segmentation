import cv2
import numpy as np
import matplotlib.image as mpimg
import scipy.stats
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
from numpy.random import randint, random

# input fllename >> output 3d array
from utilities.Path_Join import p_join
from Traitement_image.Preprocessing_for_cropped import preproc_and_crop


def read_img(filename: object, size: object) -> object:
    img_3d = mpimg.imread(filename)
    # Downsample the image
    small = cv2.resize(img_3d, (0, 0), fx=size[0], fy=size[1])
    return small


# input 3d array >> output 2d array
def flatten_img(img_3d):
    x, y = img_3d.shape
    img_2d = img_3d.reshape(x * y, 1)
    img_2d = np.array(img_2d, dtype=np.float64)
    return img_2d


# input 2d array >> output 3d array
def recover_img(img_2d, x, y):
    img_2d = cv2.resize(img_2d, (0, 0), fx=10, fy=10)
    recover_image = img_2d.reshape(x, y)
    return recover_image


def random_init(img, k):
    # For gray-scale
    if len(img.shape) == 1:
        means = randint(low=0, high=255, size=(k, 1))
        cov = randint(low=0, high=500, size=k)
        pis = random(size=k)
    else:
        z = img.shape[1]
        means = randint(low=0, high=255, size=(k, z))
        cov = randint(low=0, high=500, size=(k, z, z))
        pis = random(size=k)
    return means, cov, pis


# E-Step: Update Parameters
# update the conditional pdf - prob that pixel i given class j
def update_responsibility(img, means, cov, pis, k):
    # responsibilities: i th pixels, j th class
    # pis * gaussian.pdf
    responsibilities = np.array(
        [pis[j] * scipy.stats.multivariate_normal.pdf(img, mean=means[j], cov=cov[j]) for j in range(k)]).T
    # normalize for each row
    norm = np.sum(responsibilities, axis=1)
    # convert to column vector
    norm = np.reshape(norm, (len(norm), 1))
    responsibilities = responsibilities / norm
    return responsibilities


# update pi for each class of Gaussian model
def update_pis(responsibilities):
    pis = np.sum(responsibilities, axis=0) / responsibilities.shape[0]
    return pis


# update means for each class of Gaussian model
def update_means(img, responsibilities):
    means = []
    class_n = responsibilities.shape[1]
    for j in range(class_n):
        weight = responsibilities[:, j] / np.sum(responsibilities[:, j])
        weight = np.reshape(weight, (1, len(weight)))
        means_j = weight.dot(img)
        means.append(means_j[0])
    means = np.array(means)
    return means


# update covariance matrix for each class of Gaussian model
def update_covariance(img, responsibilities, means):
    cov = []
    class_n = responsibilities.shape[1]
    for j in range(class_n):
        weight = responsibilities[:, j] / np.sum(responsibilities[:, j])
        weight = np.reshape(weight, (1, len(weight)))
        # Each pixels have a covariance matrice
        covs = [np.mat(i - means[j]).T * np.mat(i - means[j]) for i in img]
        # Weighted sum of covariance matrices
        cov_j = sum(weight[0][i] * covs[i] for i in range(len(weight[0])))
        cov.append(cov_j)
    cov = np.array(cov)
    return cov


# M-step: choose a label that maximise the likelihood
def update_labels(responsibilities):
    labels = np.argmax(responsibilities, axis=1)
    return labels


def update_loglikelihood(img, means, cov, pis, k):
    pdf = np.array([pis[j] * scipy.stats.multivariate_normal.pdf(img, mean=means[j], cov=cov[j]) for j in range(k)])
    log_ll = np.log(np.sum(pdf, axis=0))
    log_ll_sum = np.sum(log_ll)
    return log_ll_sum


def em_cluster(img, k, error=10e-3, iter_n=1000):
    #  init setting
    start_dt1 = datetime.datetime.now()
    cnt = 0
    likelihood_arr = []
    means_arr = []
    means, cov, pis = random_init(img, k)
    likelihood = 0
    new_likelihood = 2
    means_arr.append(means)
    responsibilities = update_responsibility(img, means, cov, pis, k)
    while (abs(likelihood - new_likelihood) > error) and (cnt != iter_n):
        start_dt = datetime.datetime.now()
        cnt += 1
        likelihood = new_likelihood
        # M-Step
        labels = update_labels(responsibilities)
        # E-step
        responsibilities = update_responsibility(img, means, cov, pis, k)
        means = update_means(img, responsibilities)
        cov = update_covariance(img, responsibilities, means)
        pis = update_pis(responsibilities)
        new_likelihood = update_loglikelihood(img, means, cov, pis, k)
        likelihood_arr.append(new_likelihood)
        end_dt = datetime.datetime.now()
        diff = relativedelta(end_dt, start_dt)
        print("iter: %s, time interval: %s:%s:%s:%s" % (cnt, diff.hours, diff.minutes, diff.seconds, diff.microseconds))
        print("log-likelihood = {}".format(new_likelihood))
        # Store means stat
        means_arr.append(means)
    likelihood_arr = np.array(likelihood_arr)
    print('Converge at iteration {}'.format(cnt + 1))
    end_dt1 = datetime.datetime.now()
    diff = relativedelta(end_dt1, start_dt1)
    print("duration time: %s:%s:%s:%s" % (diff.hours, diff.minutes, diff.seconds, diff.microseconds))
    return labels, means, cov, pis, likelihood_arr, means_arr


def prepare_image(excel_path, lign_number,
                  image_path_prepro=r'C:\Users\Asus\PycharmProjects\pythonProject1\temp\prepoc_img.jpg',
                  dataset_directory=r'C:\Users\Asus\Desktop\PCD\Organized dataset\curated'):
    df = pd.read_excel(excel_path)
    number = lign_number
    center = eval(df['center of roi'][number])
    shape = eval(df['Shape of cropped image'][number])
    image_path_raw = p_join(dataset_directory, df['image full mammo path'][number])
    raw = mpimg.imread(image_path_raw)
    prepro = preproc_and_crop(raw, center, shape)
    cv2.imwrite(image_path_prepro, prepro)
    return image_path_prepro


def aply_em(image_path_prepro):
    gray_img = read_img(filename=image_path_prepro, size=(0.5, 0.5))
    x, y = gray_img.shape
    dim = (x, y)
    flat_im = flatten_img(gray_img)
    labels, means, cov, pis, likelihood_arr, means_arr = em_cluster(flat_im, 2, error=0.001)
    means = np.array([element[0] for element in means])
    em_img = means[labels]
    recover_imag = em_img.reshape(dim[0], dim[1])
    return recover_imag
