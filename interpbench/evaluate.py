"""
evaluate.py

InterpBench evaluation engine.

evaluate() benchmarks a PyTorch network and returns the canonical
InterpBench Report object containing a single Snapshot.

The evaluation follows a two-pass streaming design:

Pass 1
------
    • Generate adversarial examples
    • Extract intermediate activations through hooks
    • Fit StreamingPCA for every layer

Pass 2
------
    • Generate adversarial examples
    • Extract intermediate activations
    • Transform activations into PCA space
    • Update GeometryBackend for every
      (predicted class, true class) pair

Finalization
------------
    • Compute convex hull volumes
    • Compute Score() and Integral()
    • Create LayerReport objects

This implementation evaluates the entire test set rather than a
single batch.
"""
from __future__ import annotations
import os
import psutil

process = psutil.Process(os.getpid())

def print_ram(tag):
    rss = process.memory_info().rss / (1024 ** 3)
    print(f"[RAM] {tag:<30} {rss:6.2f} GB", flush=True)


import numpy as np
import torch

from mair.attacks import PGD

from .report import (
    Report,
    Snapshot,
    LayerReport,
)

from .metrics import (
    StreamingPCA,
    HullBackend,
    Score,
    Integral,
)


# ---------------------------------------------------------------------
# Feature extraction using forward hooks
# ---------------------------------------------------------------------


def register_hooks(model):
    """
    Register forward hooks on benchmark layers.

    Returns
    -------
    activations : dict
        Dictionary populated during forward passes.

    handles : list
        Hook handles to remove later.
    """

    activations = {}
    handles = []

    counter = 0

    for name, module in model.model.named_children():

        if "layer" in name or "block" in name:

            layer_name = f"feat{counter}"

            counter += 1


            def hook_factory(key):

                def hook(module, inputs, output):

                    # Some models return tuples/lists (e.g. auxiliary outputs)
                    if isinstance(output, (tuple, list)):
                        output = output[0]

                    # CNN feature maps: (N, C, H, W) -> (N, C)
                    if output.ndim == 4:
                        output = output.mean(dim=(2, 3))

                    # Transformer features: (N, Tokens, D) -> (N, D)
                    elif output.ndim == 3:
                        output = output.mean(dim=1)

                    # Move compact activations to CPU
                    # MEMORY OPTIMIZATION:
                    # Move activations away from GPU immediately.
                    activations[key] = (
                        output
                        .detach()
                        .cpu()
                    )

                return hook


            handle = module.register_forward_hook(
                hook_factory(layer_name)
            )

            handles.append(handle)

    return activations, handles



def remove_hooks(handles):
    """
    Remove forward hooks.
    """

    for handle in handles:
        handle.remove()



# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def infer_num_classes(testloader):
    """
    Infer number of classes when possible.
    """

    dataset = testloader.dataset

    if hasattr(dataset, "classes"):

        return len(dataset.classes)

    return 10



def successful_indices(
    prediction,
    labels,
):
    """
    Return indices of successful adversarial attacks.
    """

    mask = prediction != labels

    return torch.where(mask)[0]



def cleanup_batch():

    """
    Release temporary GPU memory.
    """

    if torch.cuda.is_available():

        torch.cuda.empty_cache()



# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------


def evaluate(
    model,
    testloader,
    *,
    model_name=None,
    dataset=None,
    trainer=None,
    run=None,
    epoch=None,
):

    device = next(
        model.parameters()
    ).device


    # ---------------------------------------------------------------
    # Standard benchmark metrics
    # ---------------------------------------------------------------

    accuracy = model.eval_accuracy(
        testloader
    )

    fgsm = None
    pgd = None


    try:

        fgsm = model.eval_rob_accuracy_fgsm(
            testloader,
            eps=8 / 255,
            verbose=False,
        )

    except Exception:

        pass


    try:

        pgd = model.eval_rob_accuracy_pgd(
            testloader,
            eps=8 / 255,
            alpha=2 / 255,
            steps=10,
            verbose=False,
        )

    except Exception:

        pass



    # ---------------------------------------------------------------
    # Feature extraction setup
    # ---------------------------------------------------------------

    activations, handles = register_hooks(
        model
    )


    layer_names = [
        f"feat{i}"
        for i in range(len(handles))
    ]


    num_classes = infer_num_classes(
        testloader
    )


    # ---------------------------------------------------------------
    # PASS 1
    #
    # Fit PCA models
    # ---------------------------------------------------------------

    pca_models = {

        layer:
        StreamingPCA(
            n_components=10
        )

        for layer in layer_names

    }


    attack = PGD(model)

    model.eval()

    # ---------------------------------------------------------------
    # Build adversarial cache
    #
    # Generate adversarial examples only once. They are valid for the
    # current model because the network remains fixed throughout this
    # evaluation.
    # ---------------------------------------------------------------

    adv_batches = []

    for images, labels in testloader:

        images = images.to(device)
        labels = labels.to(device)

        adv_images = attack(
            images,
            labels,
        )

        adv_batches.append(
            (
                adv_images.detach().cpu(),
                labels.detach().cpu(),
            )
        )

        del images
        del labels
        del adv_images

        cleanup_batch()

    for batch_idx, (adv_images, labels) in enumerate(adv_batches):

        adv_images = adv_images.to(device)
        labels = labels.to(device)

        with torch.no_grad():

            logits = model(
                adv_images
            )

        prediction = torch.argmax(
            logits,
            dim=1,
        )


        successful = successful_indices(
            prediction,
            labels,
        )


        if len(successful) > 0:

            successful_cpu = (
                successful
                .cpu()
            )


            for layer_name in layer_names:

                if layer_name not in activations:
                    continue


                features = (
                    activations[layer_name]
                    [successful_cpu]
                )


                pca_models[layer_name].partial_fit(
                    features
                )

                del features


        del adv_images
        del logits
        del prediction
        del successful
        activations.clear()
        cleanup_batch()



    # ---------------------------------------------------------------
    # PASS 2
    #
    # PCA transform + geometry accumulation
    # ---------------------------------------------------------------


    geometry = {

        layer:
        [
            [
                HullBackend()
                for _ in range(num_classes)
            ]

            for _ in range(num_classes)
        ]

        for layer in layer_names

    }

    for batch_idx, (adv_images, labels) in enumerate(adv_batches):

        adv_images = adv_images.to(device)
        labels = labels.to(device)

        with torch.no_grad():

            logits = model(
                adv_images
            )

        prediction = torch.argmax(
            logits,
            dim=1,
        )


        successful = successful_indices(
            prediction,
            labels,
        )


        if len(successful) == 0:

            activations.clear()
            cleanup_batch()

            continue


        successful_cpu = (
            successful
            .cpu()
        )


        pred_success = (
            prediction[successful]
            .cpu()
        )

        label_success = (
            labels[successful]
            .cpu()
        )



        for layer_name in layer_names:


            features = (
                activations[layer_name]
                [successful_cpu]
            )


            projected = (
                pca_models[layer_name]
                .transform(features)
            )


            # MEMORY OPTIMIZATION:
            # Update geometry in batches rather than one point at a time.

            for pred_class in range(num_classes):

                for true_class in range(num_classes):

                    mask = (

                        (pred_success == pred_class)
                        &
                        (label_success == true_class)

                    )


                    if mask.any():

                        geometry[layer_name][
                            pred_class
                        ][
                            true_class
                        ].update(

                            projected[
                                mask.numpy()
                            ]

                        )
            del features
            del projected


        del pred_success
        del label_success

        del adv_images
        del logits
        del prediction
        del successful

        activations.clear()
        cleanup_batch()
        
    # ---------------------------------------------------------------
    # Finalize geometry
    # ---------------------------------------------------------------

    layer_reports = []


    print("\nInterpretability Statistics")
    print("-" * 100)

    print(
        f"{'Layer':<12}"
        f"{'Interp Score':>30}"
        f"{'Interp Integral':>30}"
        f"{'Variance kept':>30}"
    )

    print("-" * 100)



    for layer_name in layer_names:


        matrix = np.zeros(
            (
                num_classes,
                num_classes,
            )
        )


        for i in range(num_classes):

            for j in range(num_classes):

                result = geometry[layer_name][i][j].finalize()

                matrix[i, j] = result.volume



        score = Score(
            matrix
        )

        integral = Integral(
            matrix
        )


        layer_reports.append(

            LayerReport(

                layer=layer_name,

                matrix=matrix,

                score=score,

                integral=integral,

            )

        )


        print(

            f"{layer_name:<12}"
            f"{score:>30.6e}"
            f"{integral:>30.6e}"
            f"{100*pca_models[layer_name].variance_retained:>29.2f}%"

        )


    print("-" * 100)



    remove_hooks(
        handles
    )
    
    del adv_batches
    cleanup_batch()



    # ---------------------------------------------------------------
    # Snapshot
    # ---------------------------------------------------------------


    mean_score = float(
        np.mean(
            [
                layer.score
                for layer in layer_reports
            ]
        )
    )


    mean_integral = float(
        np.mean(
            [
                layer.integral
                for layer in layer_reports
            ]
        )
    )


    snapshot = Snapshot(

        model=model_name,

        dataset=dataset,

        trainer=trainer,

        run=run,

        epoch=epoch,

        accuracy=accuracy,

        fgsm_accuracy=fgsm,

        pgd_accuracy=pgd,

        interpretability_score=mean_score,

        interpretability_integral=mean_integral,

        layers=layer_reports,

    )


    report = Report()

    report.add(
        snapshot
    )


    return report