"""Shared library of gate templates and tileable component primitives.

This directory is CPU-independent. Any CPU project can use these
templates to generate N-bit registers, ALUs, etc.

Structure:
    lib/gates/          — NAND, NOR, INV .asc schematics (2-4 input variants)
    lib/components/     — tileable component templates (register, invertible_register)
"""

import os

LIB_DIR = os.path.dirname(os.path.abspath(__file__))
GATES_DIR = os.path.join(LIB_DIR, "gates")
COMPONENTS_DIR = os.path.join(LIB_DIR, "components")


def gate_path(name):
    """Get path to a gate .asc file. e.g. gate_path('NAND') or gate_path('3NOR')."""
    return os.path.join(GATES_DIR, f"{name}.asc")


def component_path(component, filename):
    """Get path to a component template. e.g. component_path('register', '1bit.asc')."""
    return os.path.join(COMPONENTS_DIR, component, filename)


def register_paths():
    """Return (one, two, three) paths for register tiling."""
    d = os.path.join(COMPONENTS_DIR, "register")
    return (
        os.path.join(d, "1bit.asc"),
        os.path.join(d, "tile.asc"),
        os.path.join(d, "tile_offset.asc"),
    )


def invertible_register_paths():
    """Return register paths + xor_template for invertible register tiling."""
    reg = register_paths()
    xor = os.path.join(COMPONENTS_DIR, "invertible_register", "xor_template.asc")
    return reg + (xor,)
