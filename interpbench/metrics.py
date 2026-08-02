"""
metrics.py

Numerical algorithms used by InterpBench.

This module contains the numerical backend for interpretability
evaluation. It is intentionally independent of the evaluation
pipeline.

Architecture
------------

StreamingPCA
    Learns a global PCA basis incrementally over the benchmark
    dataset and projects activations into that basis.

GeometryBackend
    Abstract interface for geometric statistics computed on the
    projected point cloud.

HullBackend
    Current GeometryBackend implementation based on convex hull
    volume.

HullResult
    Geometry-only result returned by GeometryBackend.finalize().

evaluate.py is responsible for orchestrating the benchmark and
combining PCA statistics with HullResult.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from scipy.spatial import ConvexHull
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler



# ---------------------------------------------------------------------
# Benchmark metrics
# ---------------------------------------------------------------------


def Score(matrix: np.ndarray) -> float:
    """
    Compute the InterpBench Score.

    Lower geometric interference produces a larger score.
    """

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    if np.count_nonzero(matrix) == 0:

        return 1.0


    return float(
        1.0
        /
        (
            np.count_nonzero(matrix)
            *
            np.mean(matrix)
            +
            1.0
        )
    )



def Integral(matrix: np.ndarray) -> float:
    """
    Compute the InterpBench Integral.

    Smaller total interference produces a larger value.
    """

    matrix = np.asarray(
        matrix,
        dtype=float,
    )


    return float(
        1.0
        /
        (
            np.sum(matrix)
            +
            1.0
        )
    )



# ---------------------------------------------------------------------
# Optional surface metric
# ---------------------------------------------------------------------


def area(a, b, c):

    ab = torch.sub(
        b,
        a,
    )

    ac = torch.sub(
        c,
        a,
    )

    return torch.linalg.norm(
        ab * ac
    ).item()



def surface(points):
    """
    Surface approximation retained from the original implementation.
    """

    s = 0.0

    a = points[0]

    b = points[1]

    triangles = points[2:]


    if len(triangles) == 3:

        return area(
            triangles[0],
            triangles[1],
            triangles[2],
        )


    while len(triangles) > 3:

        c = triangles[0]

        triangles = triangles[1:]

        s += area(
            a,
            b,
            c,
        )


    return s



# ---------------------------------------------------------------------
# Streaming PCA
# ---------------------------------------------------------------------


class StreamingPCA:
    """
    Incremental PCA model.

    Workflow
    --------

    Pass 1

        partial_fit(batch)

    Pass 2

        projected = transform(batch)

    evaluate.py owns the StreamingPCA instance.
    """

    def __init__(
        self,
        n_components: int = 10,
        standardize: bool = True,
    ):

        self.standardize = standardize


        self.scaler = (

            StandardScaler()

            if standardize

            else None

        )


        self.pca = IncrementalPCA(
            n_components=n_components
        )


        self._fitted = False



    def partial_fit(
        self,
        activations: torch.Tensor,
    ):
        """
        Update PCA basis using one minibatch.
        """


        x = (

            activations

            .flatten(
                start_dim=1
            )

            .detach()

            .cpu()

            .numpy()

            .astype(
                np.float32,
                copy=False,
            )

        )


        if self.scaler is not None:

            self.scaler.partial_fit(
                x
            )

            x = self.scaler.transform(
                x
            ).astype(
                np.float32,
                copy=False,
            )


        self.pca.partial_fit(
            x
        )


        self._fitted = True




    def transform(
        self,
        activations: torch.Tensor,
    ) -> np.ndarray:
        """
        Project activations into PCA space.
        """


        if not self._fitted:

            raise RuntimeError(
                "StreamingPCA must be fitted before transform()."
            )


        x = (

            activations

            .flatten(
                start_dim=1
            )

            .detach()

            .cpu()

            .numpy()

            .astype(
                np.float32,
                copy=False,
            )

        )


        if self.scaler is not None:

            x = self.scaler.transform(
                x
            ).astype(
                np.float32,
                copy=False,
            )


        return self.pca.transform(
            x
        ).astype(
            np.float32,
            copy=False,
        )



    @property
    def variance_retained(self) -> float:

        if not self._fitted:

            return 0.0


        return float(
            np.sum(
                self.pca.explained_variance_ratio_
            )
        )



    @property
    def pca_dimension(self) -> int:

        if not self._fitted:

            return 0


        return int(
            self.pca.n_components_
        )



# ---------------------------------------------------------------------
# Geometry results
# ---------------------------------------------------------------------


@dataclass
class HullResult:
    """
    Geometry-only result.

    PCA statistics intentionally do not belong here.
    """

    volume: float

    num_examples: int

    num_vertices: int

    vertex_fraction: float

    boundary_measure: float

    failed: bool

    reason: str | None



# ---------------------------------------------------------------------
# Geometry backend
# ---------------------------------------------------------------------


class GeometryBackend(ABC):

    """
    Abstract geometry backend.
    """

    @abstractmethod
    def update(
        self,
        points: np.ndarray,
    ):

        pass



    @abstractmethod
    def finalize(
        self,
    ) -> HullResult:

        pass



# ---------------------------------------------------------------------
# Convex hull backend
# ---------------------------------------------------------------------


class HullBackend(GeometryBackend):
    """
    Memory-controlled convex hull backend.

    Stores only a bounded sample of PCA points.
    This prevents geometry accumulation from exceeding RAM.
    """


    def __init__(
        self,
        max_points: int = 200,
    ):

        self.max_points = max_points

        self._points = []

        self._count = 0



    def update(
        self,
        points: np.ndarray,
    ):

        if points is None:

            return


        points = np.asarray(
            points,
            dtype=np.float32,
        )


        if len(points) == 0:

            return


        self._count += len(points)


        self._points.append(
            points
        )


        current_size = sum(
            len(x)
            for x in self._points
        )


        if current_size > self.max_points:


            merged = np.concatenate(
                self._points,
                axis=0,
            )


            indices = np.random.choice(
                len(merged),
                self.max_points,
                replace=False,
            )


            self._points = [

                merged[indices]

            ]



    def finalize(
        self,
    ) -> HullResult:


        if len(self._points) == 0:

            return HullResult(

                volume=0.0,

                num_examples=0,

                num_vertices=0,

                vertex_fraction=0.0,

                boundary_measure=0.0,

                failed=True,

                reason="No points received.",

            )


        points = np.concatenate(
            self._points,
            axis=0,
        )


        num_examples = self._count



        if (

            points.ndim != 2

            or len(points) < 3

            or points.shape[1] < 2

        ):

            return HullResult(

                volume=0.0,

                num_examples=num_examples,

                num_vertices=0,

                vertex_fraction=0.0,

                boundary_measure=0.0,

                failed=True,

                reason="Insufficient independent samples.",

            )



        try:


            hull = ConvexHull(
                points
            )


            num_vertices = len(
                hull.vertices
            )


            return HullResult(

                volume=float(
                    hull.volume
                ),

                num_examples=num_examples,

                num_vertices=num_vertices,

                vertex_fraction=(

                    num_vertices

                    /

                    num_examples

                ),

                boundary_measure=float(
                    hull.area
                ),

                failed=False,

                reason=None,

            )



        except Exception as e:


            return HullResult(

                volume=0.0,

                num_examples=num_examples,

                num_vertices=0,

                vertex_fraction=0.0,

                boundary_measure=0.0,

                failed=True,

                reason=str(e),

            )