#!/usr/bin/env python3
from __future__ import annotations

"""
Shared geometry ontology for evaluation.

This module contains only domain/schema definitions shared by:

- accuracy.py
- anchor.py
- represent.py

It intentionally contains no:
- dataset paths
- checkpoint paths
- output paths
- PyTorch/model code
- evaluation hyperparameters

Definitions
-----------
1. Stage 1 primitive concepts
2. Stage 2 relation/property concepts
3. Loose free-form aliases
4. Stage 2 anchor DSL fields
5. Stage 2 -> Stage 1 primitive-parent ontology
6. Semantic JSON field semantics used for exact matching
"""

from typing import Final


# =============================================================================
# Stage 1 primitive concepts
# =============================================================================

STAGE1_CONCEPTS: Final[tuple[str, ...]] = (
    "point",
    "segment",
    "angle",
    "triangle",
    "quadrilateral",
    "circle",
)


# =============================================================================
# Stage 2 relation / property concepts
# =============================================================================

STAGE2_CONCEPTS: Final[tuple[str, ...]] = (
    "triangle_altitude",
    "triangle_median",
    "triangle_angle_bisector",
    "triangle_perpendicular_bisector",
    "triangle_centroid",
    "triangle_circumcenter",
    "triangle_incenter",
    "isosceles_triangle",
    "right_triangle",
    "equilateral_triangle",
    "corresponding_angles",
    "alternate_interior_angles",
    "same_side_interior_angles",
    "vertical_angles",
    "parallelogram",
    "rectangle",
    "rhombus",
    "square",
    "trapezoid",
    "circle_radius",
    "circle_diameter",
    "circle_chord",
    "circle_sector",
    "circle_tangent",
    "central_angle",
    "inscribed_angle",
    "cyclic_quadrilateral",
)


ALL_CONCEPTS: Final[tuple[str, ...]] = (
    *STAGE1_CONCEPTS,
    *STAGE2_CONCEPTS,
)


# =============================================================================
# Loose scorer aliases
# =============================================================================
#
# IMPORTANT:
# These aliases are ONLY for the legacy/free-form Loose Accuracy scorer.
#
# They must NOT be used for:
# - constrained classification
# - semantic exact scoring
# - representation evaluation
#
# The canonical concept name itself is added separately by the scorer as:
#
#     concept.replace("_", " ")
#
# Therefore this dictionary contains natural-language alternatives that the
# free-form model may reasonably produce.
#
# Broad aliases such as "radius" and "median" are intentionally tolerated by
# the Loose scorer. This makes Loose Accuracy permissive, which is why it is
# diagnostic/legacy rather than the primary concept metric.
# =============================================================================

CONCEPT_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    # -------------------------------------------------------------------------
    # Stage 1
    # -------------------------------------------------------------------------
    "point": (
        "point",
    ),

    "segment": (
        "line segment",
        "segment",
    ),

    "angle": (
        "angle",
    ),

    "triangle": (
        "triangle",
    ),

    "quadrilateral": (
        "quadrilateral",
        "quadrangle",
    ),

    "circle": (
        "circle",
    ),

    # -------------------------------------------------------------------------
    # Stage 2: triangle constructions / centers
    # -------------------------------------------------------------------------
    "triangle_altitude": (
        "triangle altitude",
        "altitude",
    ),

    "triangle_median": (
        "triangle median",
        "median",
    ),

    "triangle_angle_bisector": (
        "triangle angle bisector",
        "angle bisector",
        "bisects angle",
    ),

    "triangle_perpendicular_bisector": (
        "triangle perpendicular bisector",
        "perpendicular bisector",
    ),

    "triangle_centroid": (
        "triangle centroid",
        "centroid",
    ),

    "triangle_circumcenter": (
        "triangle circumcenter",
        "circumcenter",
    ),

    "triangle_incenter": (
        "triangle incenter",
        "incenter",
    ),

    # -------------------------------------------------------------------------
    # Stage 2: triangle properties
    # -------------------------------------------------------------------------
    "isosceles_triangle": (
        "isosceles triangle",
        "isosceles",
    ),

    "right_triangle": (
        "right triangle",
    ),

    "equilateral_triangle": (
        "equilateral triangle",
        "equilateral",
    ),

    # -------------------------------------------------------------------------
    # Stage 2: angle relations
    # -------------------------------------------------------------------------
    "corresponding_angles": (
        "corresponding angles",
        "corresponding angle",
    ),

    "alternate_interior_angles": (
        "alternate interior angles",
        "alternate interior angle",
    ),

    "same_side_interior_angles": (
        "same side interior angles",
        "same-side interior angles",
        "same side interior angle",
    ),

    "vertical_angles": (
        "vertical angles",
        "vertical angle",
        "vertically opposite angles",
    ),

    # -------------------------------------------------------------------------
    # Stage 2: quadrilaterals
    # -------------------------------------------------------------------------
    "parallelogram": (
        "parallelogram",
    ),

    "rectangle": (
        "rectangle",
    ),

    "rhombus": (
        "rhombus",
    ),

    "square": (
        "square",
    ),

    "trapezoid": (
        "trapezoid",
        "trapezium",
    ),

    # -------------------------------------------------------------------------
    # Stage 2: circle relations / parts
    # -------------------------------------------------------------------------
    "circle_radius": (
        "circle radius",
        "radius",
    ),

    "circle_diameter": (
        "circle diameter",
        "diameter",
    ),

    "circle_chord": (
        "circle chord",
        "chord",
    ),

    "circle_sector": (
        "circle sector",
        "sector",
    ),

    "circle_tangent": (
        "circle tangent",
        "tangent",
    ),

    "central_angle": (
        "central angle",
    ),

    "inscribed_angle": (
        "inscribed angle",
    ),

    "cyclic_quadrilateral": (
        "cyclic quadrilateral",
        "inscribed quadrilateral",
    ),
}


# =============================================================================
# Stage 2 Anchor DSL
# =============================================================================
#
# These are the native top-level serialization fields emitted by the final
# Stage 2 generator.
#
# They are fact TYPES, not Stage 1 primitives and not Stage 2 concepts.
#
# Example:
#
# concept = triangle_median
#
# POINTS: A B C D
# SEG: AB AC BC AD
# ON: ON(D,BC)
# EQ: TICK(BD,1) TICK(CD,1)
#
# =============================================================================

ANCHOR_DSL_FIELDS: Final[tuple[str, ...]] = (
    "POINTS",
    "SEG",
    "CIRCLE",
    "CENTER",
    "PERP",
    "EQ",
    "PARA",
    "ANG",
    "SECTOR",
    "ON",
)


# =============================================================================
# Stage 2 -> Stage 1 primitive-parent ontology
# =============================================================================
#
# Used by representation.py for Parent Margin:
#
#     parent_similarity
#     -
#     non_parent_similarity
#
# Every listed parent MUST be one of STAGE1_CONCEPTS.
#
# There is intentionally no "parent-group" definition here.
# Parent-group separation was removed because broad parents such as segment
# and angle caused unrelated Stage 2 concepts to be grouped together.
#
# Each Stage 2 concept may have multiple relevant primitive parents.
# =============================================================================

PARENT_ONTOLOGY: Final[dict[str, tuple[str, ...]]] = {
    # -------------------------------------------------------------------------
    # Triangle constructions
    # -------------------------------------------------------------------------
    "triangle_altitude": (
        "triangle",
        "segment",
        "angle",
    ),

    "triangle_median": (
        "triangle",
        "segment",
        "point",
    ),

    "triangle_angle_bisector": (
        "triangle",
        "angle",
        "segment",
    ),

    "triangle_perpendicular_bisector": (
        "triangle",
        "segment",
        "angle",
    ),

    # -------------------------------------------------------------------------
    # Triangle centers
    # -------------------------------------------------------------------------
    "triangle_centroid": (
        "triangle",
        "point",
    ),

    "triangle_circumcenter": (
        "triangle",
        "point",
    ),

    "triangle_incenter": (
        "triangle",
        "point",
    ),

    # -------------------------------------------------------------------------
    # Triangle properties
    # -------------------------------------------------------------------------
    "isosceles_triangle": (
        "triangle",
        "segment",
    ),

    "right_triangle": (
        "triangle",
        "angle",
    ),

    "equilateral_triangle": (
        "triangle",
        "segment",
    ),

    # -------------------------------------------------------------------------
    # Angle relations
    # -------------------------------------------------------------------------
    "corresponding_angles": (
        "angle",
        "segment",
    ),

    "alternate_interior_angles": (
        "angle",
        "segment",
    ),

    "same_side_interior_angles": (
        "angle",
        "segment",
    ),

    "vertical_angles": (
        "angle",
        "segment",
    ),

    # -------------------------------------------------------------------------
    # Quadrilateral properties
    # -------------------------------------------------------------------------
    "parallelogram": (
        "quadrilateral",
        "segment",
    ),

    "rectangle": (
        "quadrilateral",
        "angle",
    ),

    "rhombus": (
        "quadrilateral",
        "segment",
    ),

    "square": (
        "quadrilateral",
        "segment",
        "angle",
    ),

    "trapezoid": (
        "quadrilateral",
        "segment",
    ),

    # -------------------------------------------------------------------------
    # Circle parts / relations
    # -------------------------------------------------------------------------
    "circle_radius": (
        "circle",
        "segment",
    ),

    "circle_diameter": (
        "circle",
        "segment",
    ),

    "circle_chord": (
        "circle",
        "segment",
    ),

    "circle_sector": (
        "circle",
        "angle",
    ),

    "circle_tangent": (
        "circle",
        "segment",
    ),

    "central_angle": (
        "circle",
        "angle",
    ),

    "inscribed_angle": (
        "circle",
        "angle",
    ),

    "cyclic_quadrilateral": (
        "circle",
        "quadrilateral",
    ),
}


# =============================================================================
# Semantic Exact JSON field semantics
# =============================================================================
#
# These definitions are used by accuracy.py when comparing a model-generated
# entity JSON object with semantic_target_json.
#
# They specify which ordering differences are semantically irrelevant.
#
# Example:
#
#     segment AB == BA
#
# and
#
#     angle [A, B, C] == [C, B, A]
#
# where B remains the vertex.
# =============================================================================

SEMANTIC_UNDIRECTED_SEGMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "segment",
        "whole_segment",
        "altitude_segment",
        "median_segment",
        "bisector_segment",
        "bisector",
        "bisected_segment",
        "radius_segment",
        "diameter_segment",
        "chord_segment",
        "tangent_segment",
    }
)


SEMANTIC_UNORDERED_STRING_LIST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "points",
        "endpoints",
        "vertices",
        "triangle",
        "quadrilateral",
        "arms",
        "boundary_points",
        "angle_markers",
    }
)


SEMANTIC_UNORDERED_NESTED_LIST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "sides",
        "segments",
        "subsegments",
        "rays",
    }
)


# =============================================================================
# Ontology validation
# =============================================================================

def _validate_ontology() -> None:
    stage1 = set(STAGE1_CONCEPTS)
    stage2 = set(STAGE2_CONCEPTS)
    all_concepts = stage1 | stage2

    # -------------------------------------------------------------------------
    # Concept sets
    # -------------------------------------------------------------------------
    if len(stage1) != len(STAGE1_CONCEPTS):
        raise RuntimeError(
            "STAGE1_CONCEPTS contains duplicate entries."
        )

    if len(stage2) != len(STAGE2_CONCEPTS):
        raise RuntimeError(
            "STAGE2_CONCEPTS contains duplicate entries."
        )

    overlap = stage1 & stage2
    if overlap:
        raise RuntimeError(
            "Stage 1 and Stage 2 concept ontologies overlap: "
            f"{sorted(overlap)}"
        )

    # -------------------------------------------------------------------------
    # Alias coverage
    # -------------------------------------------------------------------------
    alias_keys = set(CONCEPT_ALIASES)

    missing_aliases = all_concepts - alias_keys
    extra_aliases = alias_keys - all_concepts

    if missing_aliases:
        raise RuntimeError(
            "CONCEPT_ALIASES is missing concept(s): "
            f"{sorted(missing_aliases)}"
        )

    if extra_aliases:
        raise RuntimeError(
            "CONCEPT_ALIASES contains unknown concept(s): "
            f"{sorted(extra_aliases)}"
        )

    for concept, aliases in CONCEPT_ALIASES.items():
        if not aliases:
            raise RuntimeError(
                f"CONCEPT_ALIASES[{concept!r}] is empty."
            )

        if len(set(aliases)) != len(aliases):
            raise RuntimeError(
                f"CONCEPT_ALIASES[{concept!r}] contains duplicates."
            )

    # -------------------------------------------------------------------------
    # Parent ontology coverage
    # -------------------------------------------------------------------------
    parent_keys = set(PARENT_ONTOLOGY)

    missing_parent_definitions = stage2 - parent_keys
    extra_parent_definitions = parent_keys - stage2

    if missing_parent_definitions:
        raise RuntimeError(
            "PARENT_ONTOLOGY is missing Stage 2 concept(s): "
            f"{sorted(missing_parent_definitions)}"
        )

    if extra_parent_definitions:
        raise RuntimeError(
            "PARENT_ONTOLOGY contains unknown Stage 2 concept(s): "
            f"{sorted(extra_parent_definitions)}"
        )

    for concept, parents in PARENT_ONTOLOGY.items():
        if not parents:
            raise RuntimeError(
                f"PARENT_ONTOLOGY[{concept!r}] has no parent."
            )

        if len(set(parents)) != len(parents):
            raise RuntimeError(
                f"PARENT_ONTOLOGY[{concept!r}] contains duplicate parents."
            )

        unknown_parents = set(parents) - stage1

        if unknown_parents:
            raise RuntimeError(
                f"PARENT_ONTOLOGY[{concept!r}] contains "
                f"non-Stage1 parent(s): {sorted(unknown_parents)}"
            )

    # -------------------------------------------------------------------------
    # Anchor DSL
    # -------------------------------------------------------------------------
    if len(set(ANCHOR_DSL_FIELDS)) != len(ANCHOR_DSL_FIELDS):
        raise RuntimeError(
            "ANCHOR_DSL_FIELDS contains duplicate fields."
        )


_validate_ontology()
