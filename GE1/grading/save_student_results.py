import numpy as np
from matplotlib import pyplot as plt
import grading.helpers as helpers


################################################################################
# Saving student results
################################################################################
def initialize_res(scope):
    exercise_id = "sciper"
    sciper_number = helpers.resolve('sciper_number', scope)
    stud_grad = dict(sciper_number=sciper_number)
    helpers.register_answer(exercise_id, stud_grad, scope)


# ------------------------- Canny part -------------------------

def save_gaussian_kernel_1d(scope):
    exercise_id = 'gaussian_kernel_1d'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    sizes = grading_data["sizes"]
    sigmas = grading_data["sigmas"]
    
    # Apply student's function and register results
    k0 = func(sizes[0], sigmas[0])
    k1 = func(sizes[1], sigmas[1])
    k2 = func(sizes[2], sigmas[2])
    k3 = func(sizes[3], sigmas[3])
    student_res = dict(k0=k0, k1=k1, k2=k2, k3=k3)
    helpers.register_answer(exercise_id, student_res, scope)


def save_gaussian_kernel_2d(scope):
    exercise_id = 'gaussian_kernel_2d'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    k0_1d = grading_data["k0_1d"]
    k1_1d = grading_data["k1_1d"]
    k2_1d = grading_data["k2_1d"]
    k3_1d = grading_data["k3_1d"]
    
    # Apply student's function and register results
    k0 = func(k0_1d)
    k1 = func(k1_1d)
    k2 = func(k2_1d)
    k3 = func(k3_1d)
    student_res = dict(k0=k0, k1=k1, k2=k2, k3=k3)
    helpers.register_answer(exercise_id, student_res, scope)


def save_apply_conv_2d_with_stride_2(scope):
    exercise_id = 'apply_conv_2d_with_stride_2'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    img1 = grading_data["img1"]
    kernel1 = grading_data["kernel1"]
    img2 = grading_data["img2"]
    kernel2 = grading_data["kernel2"]
    
    # Apply student's function and register results
    out1 = func(img1, kernel1)
    out2 = func(img2, kernel2)
    student_res = dict(out1=out1, out2=out2)
    helpers.register_answer(exercise_id, student_res, scope)


def save_thresholding(scope):
    exercise_id = 'thresholding'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    img1 = grading_data["img1"]
    thresh1 = grading_data["thresh1"]
    img2 = grading_data["img2"]
    thresh2 = grading_data["thresh2"]
    
    # Apply student's function and register results
    out1 = func(img1, thresh1)
    out2 = func(img2, thresh2)
    student_res = dict(out1=out1, out2=out2)
    helpers.register_answer(exercise_id, student_res, scope)



# ------------------------- Non max suppression part -------------------------

def save_gaussian_blur(scope):
    exercise_id = 'gaussian_blur'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    gr_X = grading_data["img_x"]

    # Apply student's function and register results
    student_res = func(gr_X)
    helpers.register_answer(exercise_id, student_res, scope)


def save_compute_grad(scope):
    exercise_id = 'compute_grad'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    gr_X = grading_data["img_x"]

    # Apply student's function and register results
    student_res = func(gr_X)
    helpers.register_answer(exercise_id, student_res, scope)


def save_grad_direction(scope):
    exercise_id = 'grad_direction'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    gr_X = grading_data["grad_x"]
    gr_Y = grading_data["grad_y"]

    # Apply student's function and register results
    student_res = func(gr_X, gr_Y)
    helpers.register_answer(exercise_id, student_res, scope)


def save_bilinear_interpolation(scope):
    exercise_id = 'bilinear_interpolation'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    img_x = grading_data["img_x"]
    px_1 = grading_data["px_1"]
    py_1 = grading_data["py_1"]
    px_2 = grading_data["px_2"]
    py_2 = grading_data["py_2"]

    # Apply student's function and register results
    student_res_1 = func(img_x, px_1, py_1)
    student_res_2 = func(img_x, px_2, py_2)
    student_res = (student_res_1, student_res_2)
    helpers.register_answer(exercise_id, student_res, scope)


def save_find_p_r_coordinates(scope):
    exercise_id = 'find_p_r_coordinates'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    img_x = grading_data["img_x"]
    px_1 = grading_data["px_1"]
    py_1 = grading_data["py_1"]
    px_2 = grading_data["px_2"]
    py_2 = grading_data["py_2"]

    # Apply student's function and register results
    student_res_1 = func(img_x, px_1, py_1)
    student_res_2 = func(img_x, px_2, py_2)
    student_res = (student_res_1, student_res_2)
    helpers.register_answer(exercise_id, student_res, scope)


def save_is_max(scope):
    exercise_id = 'is_max'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    grad_q = grading_data["grad_q"]
    grad_p = grading_data["grad_p"]
    grad_r = grading_data["grad_r"]

    # Apply student's function and register results
    student_res = np.array([func(q,p,r) for q,p,r in zip(grad_q, grad_p, grad_r)])
    helpers.register_answer(exercise_id, student_res, scope)



# ------------------------- Hysteresis part -------------------------

def save_label_pixels(scope):
    exercise_id = 'label_pixels'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    imgs = grading_data["imgs"]
    lows = grading_data["lows"]
    highs = grading_data["highs"]

    # Apply student's function and register results
    student_res = np.array([func(img,l,h) for img,l,h in zip(imgs, lows, highs)])
    helpers.register_answer(exercise_id, student_res, scope)


def save_update(scope):
    exercise_id = 'update'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    imgs = grading_data["imgs"]
    xs = grading_data["xs"]
    ys = grading_data["ys"]

    # Apply student's function and register results
    student_res = np.array([func(img,x,y) for img,x,y in zip(imgs, xs, ys)])
    helpers.register_answer(exercise_id, student_res, scope)


def save_fast_hysteresis_thresholding(scope):
    exercise_id = 'fast_hysteresis_thresholding'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    imgs = grading_data["imgs"]
    lows = grading_data["lows"]
    highs = grading_data["highs"]

    # Apply student's function and register results
    student_res = np.array([func(img,l,h) for img,l,h in zip(imgs, lows, highs)])
    helpers.register_answer(exercise_id, student_res, scope)


def save_time_counter(scope):
    from time import sleep
    exercise_id = 'time_counter'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    laps = grading_data["laps"]
    n = grading_data["n"]

    # Apply student's function and register results
    student_res = np.array([func(sleep, t, n) for t in laps])
    helpers.register_answer(exercise_id, student_res, scope)