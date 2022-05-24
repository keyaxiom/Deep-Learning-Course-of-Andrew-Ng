#! /usr/bin/env python3

import torch
import torch.nn as nn
from torch.autograd import grad
from utils import display_progress


def grad_val_risk(imgs_val, labels_val, model, gpu=-1):
    """
    Calculates the (partial R(V))/(partial theta).

    Arguments:
        imgs_val: torch tensor, training data points. e.g. an image sample (batch_size, 3, 256, 256)
        labels_val: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_val_risk: list of torch tensor, containing the gradients from model parameters to loss on validation set.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduce='mean')

    # initialize
    if gpu >= 0:
        imgs_val, labels_val = imgs_val.cuda(), labels_val.cuda()
    outputs = model(imgs_val)
    loss = criterion(outputs, labels_val)

    # Compute sum of gradients from model parameters to loss
    params = [p for p in model.parameters() if p.requires_grad]

    grad_risk = [item.data for item in grad(loss, params)]

    return grad_risk


def s_test(val_loader, model, train_loader, gpu=1, damp=0.01, scale=25.0,
           recursion_depth=1000):
    """
    s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        val_loader: torch Dataloader, can load the validation dataset
        model: torch NN, model used to evaluate the dataset
        train_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test
    """
    # ============= Calculate derivatives of Risk_val w.r.t. model.parameters() ====================
    grad_risk = None
    print("Start calculating grad_risk...")
    for i, data in enumerate(val_loader, 0):
        images, labels = data
        if i == 0:
            grad_risk = grad_val_risk(images, labels, model, gpu=1)
        else:
            temp = grad_val_risk(images, labels, model, gpu=1)
            grad_risk = [(i + j) for i, j in zip(temp, grad_risk)]
    # ==============================================================================================
    v = [item*0.1 for item in grad_risk]
    h_estimate = v.copy()
    criterion = nn.CrossEntropyLoss(reduce='sum')
    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    print("Start estimating ...")
    for i in range(recursion_depth):
        print('Start iter %d' % (i+1))
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        #########################
        # TODO: do x, t really have to be chosen RANDOMLY from the train set?
        #########################
        for x, _, t in train_loader:
            if gpu >= 0:
                x, t = x.cuda(), t.cuda()
            y = model(x)
            loss = criterion(y, t)
            params = [p for p in model.parameters() if p.requires_grad]
            hv = hvp(loss, params, h_estimate)
            # Recursively calculate h_estimate
            h_estimate = [
                _v + (1 - damp/scale) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            break
        display_progress("Calc. s_test recursions: ", i, recursion_depth)
    return h_estimate


def hvp(y, w, v):
    """
    Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length.
    """
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First back-prop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Element-wise products
    elem_wise_products = 0.
    for grad_elem, v_elem in zip(first_grads, v):
        elem_wise_products += torch.sum(grad_elem * v_elem)

    # Second back-prop
    return_grads = [item.data for item in grad(elem_wise_products, w)]

    return return_grads
